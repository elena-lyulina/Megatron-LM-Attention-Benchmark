"""
Uniform interface over the checkpoint formats (hf / megatron).
"""

from __future__ import annotations

import math
import os
import sys
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist


class InferenceBackend(ABC):
    name: str  # "megatron" or "hf" -- used in run_metadata.json

    @abstractmethod
    def load_model(self) -> None:
        """Load the checkpoint onto this rank's device; leaves self.model in eval mode."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device the model currently lives on."""

    @abstractmethod
    def generate(self, prompt: torch.Tensor, suffix_length: int) -> torch.Tensor:
        """prompt: [B, prompt_len] -> generated: [B, suffix_length]."""

    @abstractmethod
    def forward_logits(self, inputs: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """One forward pass, no cache. Returns raw logits [B, S, V]."""

    def experiment_path_suffix(self) -> str:
        """Appended to --experiment-path so this backend/configuration's results never
        collide with another's for the same model/offset/prefix/suffix."""
        return ""

    def generate_with_capture(self, prompt: torch.Tensor, suffix_length: int,
                              prefill_callback=None, decode_step_callback=None) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} backend does not implement generate_with_capture")

    def patch_sink_scale(self) -> list:
        raise NotImplementedError(f"{self.name} backend does not implement patch_sink_scale")

    def setup_attention_capture(self, args, output_path: Path, rank: int, needs_bos: bool):
        raise NotImplementedError(f"{self.name} backend does not implement setup_attention_capture")


class MegatronBackend(InferenceBackend):
    name = "megatron"

    def __init__(self, ckpt_dir: str, tokenizer_path: str, megatron_extra_args: list | None,
                sink_scale: float | None = None):
        if not ckpt_dir or not tokenizer_path:
            raise ValueError("MegatronBackend requires ckpt_dir and tokenizer_path.")
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.megatron_extra_args = megatron_extra_args
        self.sink_scale = sink_scale
        self.model = None

    def load_model(self) -> None:
        """Load from a torch_dist checkpoint using --use-checkpoint-args. megatron_extra_args
        re-passes flags --use-checkpoint-args doesn't restore (e.g. --attention-output-gate,
        --use-rope-scaling/--rope-scaling-factor) -- sourced per model tag from MEGATRON_EXTRA
        in llama_checkpoints.sh."""
        from gpt_builders import gpt_builder
        from megatron.training import get_model
        from megatron.training.checkpointing import load_checkpoint
        from megatron.training.initialize import initialize_megatron
        from model_provider import model_provider

        saved_argv = sys.argv[:]
        sys.argv = [
            'megatron_inference',
            '--use-checkpoint-args',
            '--tensor-model-parallel-size', '1',
            '--pipeline-model-parallel-size', '1',
            '--context-parallel-size', '1',
            '--micro-batch-size', '1',
            '--global-batch-size', '4',
            '--train-iters', '1',
            '--tokenizer-type', 'HuggingFaceTokenizer',
            '--tokenizer-model', self.tokenizer_path,
            '--load', self.ckpt_dir,
            '--no-load-optim',
            '--no-load-rng',
            '--ckpt-format', 'torch_dist',
            '--dist-ckpt-strictness', 'assume_ok_unexpected',
            '--finetune',
            '--bf16',
            '--transformer-impl', 'transformer_engine',
            '--main-grads-dtype', 'fp32',
            *(self.megatron_extra_args or []),
        ]
        try:
            from megatron.training.arguments import parse_and_validate_args

            # reads arguments directly and exclusively through sys.argv -- so we're swapping
            # them beforehand. PR #4225 moved arg parsing out of initialize_megatron; launch
            # scripts must parse + set globals first.
            parse_and_validate_args()
            initialize_megatron()
            model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
            load_checkpoint(model, optimizer=None, opt_param_scheduler=None)
            self.model = model[0]
            self.model.eval()
        finally:
            sys.argv = saved_argv

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, prompt, suffix_length):
        return self.generate_with_capture(prompt, suffix_length)

    @torch.no_grad()
    def generate_with_capture(self, prompt, suffix_length, prefill_callback=None, decode_step_callback=None):
        """Greedy generation with StaticInferenceContext KV cache.

        prompt: [B, prompt_len] -- prefix tokens (BOS already included as token 0)
        prefill_callback: optional callable() invoked right after the prefill forward
                          (before any decode forward overwrites the attention buffers).
        decode_step_callback: optional callable(t: int) called after each decode step with
                       the 0-indexed step number. n_steps total = suffix_length - 1.
        Returns: [B, suffix_length] -- generated tokens
        """
        from megatron.core.inference.contexts import StaticInferenceContext
        from megatron.core.inference.utils import InferenceMode

        B, prompt_len = prompt.shape
        device = prompt.device
        max_seq_len = prompt_len + suffix_length

        ctx = StaticInferenceContext(max_batch_size=B, max_sequence_length=max_seq_len)
        ctx.reset()
        ctx.enable_prefill_mode()

        # The model only slices the logits down to the last prompt token when InferenceMode
        # is active (upstream gates this on InferenceMode.is_active(), not on inference_context
        # being set). Without it the prefill returns logits for every prompt position and we
        # would pick position 0 below, generating from the wrong token and losing all recall.
        with InferenceMode.active():
            pos = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            logits = self.model(prompt, pos, attention_mask=None, inference_context=ctx,
                               runtime_gather_output=True)
            ctx.sequence_len_offset = prompt_len
            ctx.enable_decode_mode()

            if prefill_callback is not None:
                prefill_callback()

            next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)
            generated = [next_token]

            for step_t in range(suffix_length - 1):
                pos = torch.full((B, 1), ctx.sequence_len_offset, dtype=torch.long, device=device)
                logits = self.model(next_token, pos, attention_mask=None, inference_context=ctx,
                                   runtime_gather_output=True)
                ctx.sequence_len_offset += 1
                next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                if decode_step_callback is not None:
                    decode_step_callback(step_t)

        return torch.cat(generated, dim=1)

    def forward_logits(self, inputs, position_ids):
        return self.model(inputs, position_ids, attention_mask=None)

    def experiment_path_suffix(self) -> str:
        return f"_sscale{self.sink_scale:g}" if self.sink_scale is not None else ""

    def patch_sink_scale(self) -> list:
        """Scale the virtual sink weight at inference: offset_new = offset_trained +
        log(sink_scale). sink_scale=1 is identity, >1 strengthens, <1 weakens. Supports
        off-by-one/learnable attention only. Returns original per-layer softmax_offset values."""
        from megatron.core.transformer.dot_product_attention import \
            DotProductAttention as MegatronDPA
        try:
            import transformer_engine.pytorch as te
            te_dpa_cls = te.DotProductAttention
        except ImportError:
            te_dpa_cls = None

        if self.sink_scale < 0:
            raise ValueError(f"sink_scale must be >= 0, got {self.sink_scale}")
        log_scale = math.log(self.sink_scale) if self.sink_scale > 0 else float("-inf")

        originals = []
        for module in self.model.modules():
            is_megatron_dpa = isinstance(module, MegatronDPA)
            is_te_dpa = te_dpa_cls is not None and isinstance(module, te_dpa_cls)
            if not (is_megatron_dpa or is_te_dpa) or module.softmax_offset is None:
                continue
            softmax_type = module.config.softmax_type if is_megatron_dpa else module.softmax_type
            assert softmax_type in ("off-by-one", "learnable"), (
                f"patch_sink_scale only supports off-by-one and learnable attention, got softmax_type='{softmax_type}'"
            )
            originals.append(module.softmax_offset.detach().cpu().tolist())
            module.softmax_offset.data.add_(log_scale)

        if not originals:
            raise RuntimeError("patch_sink_scale: no patchable attention layers found.")
        print(f"Patched softmax_offset += log({self.sink_scale}) = {log_scale:.4f} in {len(originals)} attention layers")
        return originals

    def setup_attention_capture(self, args, output_path: Path, rank: int, needs_bos: bool):
        """Called only when --capture-attention is set. Returns None if a previous run
        already completed the capture (jsonl generation still proceeds as needed)."""
        from attn_bench.evaluation.attn_capture import (N_BUCKETS,
                                                        AttentionCapture,
                                                        bucket_label)

        capture_marker = output_path / f"attn_scores_rouge_l_{bucket_label(N_BUCKETS - 1)}_rank{rank}.npz"
        if capture_marker.exists():
            if rank == 0:
                print("Attention capture already done -- skipping capture (jsonl still processed as needed).")
            return None

        cfg = self.model.config
        capture = AttentionCapture(
            n_layers=cfg.num_layers,
            n_heads=cfg.num_attention_heads,
            prompt_len=args.prefix_length + (1 if needs_bos else 0),
            suffix_length=args.suffix_length,
            is_gated=getattr(cfg, 'attention_output_gate', False),
        )
        capture.register(self.model)
        return capture


class HFBackend(InferenceBackend):
    name = "hf"

    def __init__(self, hf_dir: str):
        if not hf_dir:
            raise ValueError("HFBackend requires hf_dir.")
        if not Path(hf_dir).exists():
            raise FileNotFoundError(f"{hf_dir} does not exist. Run convert_and_validate_hf.slurm first.")
        self.hf_dir = hf_dir
        self.model = None

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")  # env://, reads torchrun's RANK/WORLD_SIZE/MASTER_ADDR/PORT

        self.model = AutoModelForCausalLM.from_pretrained(self.hf_dir, torch_dtype="auto").to(f"cuda:{local_rank}")
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, prompt, suffix_length):
        output = self.model.generate(
            input_ids=prompt, max_new_tokens=suffix_length, min_new_tokens=suffix_length, do_sample=False,
        )
        return output[:, prompt.shape[1]:]

    def forward_logits(self, inputs, position_ids):
        return self.model(input_ids=inputs, position_ids=position_ids, use_cache=False).logits

    def experiment_path_suffix(self) -> str:
        return "_hf"
