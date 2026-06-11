"""
Attention weight capture for decode-time visualization.

Registers a forward pre-hook on every TEDotProductAttention module (pure observation,
model output unchanged) and optionally patches _apply_output_gate for the gated model.

Captured per decode step (n_steps = suffix_length - 1; prefill at sq>1 is skipped):
  - raw attention weights w[B, np, sk] after stripping the virtual-sink column
  - sink column mass 1 - sum(w) (0 for vanilla/gated; explicit sink mass for sink/obo)
  - full-distribution entropy H(w_full) before stripping (always sums to 1, cross-model comparable)
  - gate scalar sigmoid(gate).mean(hn) [B, np] (gated model only)

Usage:
    capture = AttentionCapture(n_samples, n_layers, n_heads, n_steps, max_seq_len,
                               is_gated=False, is_rank0=True, prompt_len=500)
    capture.register(model)
    ...
    for batch in loader:
        ...
        def step_cb(t):
            capture.collect_step(t, batch_slice, is_sample_0)
        generated = greedy_generate(model, prompt, suffix_length, step_callback=step_cb)
        sample_start += B
    capture.save(inference_dir, rank)
    capture.remove()
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch


class AttentionCapture:
    def __init__(
        self,
        n_samples: int,
        n_layers: int,
        n_heads: int,
        n_steps: int,
        max_seq_len: int,
        is_gated: bool,
        is_rank0: bool,
        prompt_len: int,
    ):
        self.n_samples = n_samples
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_steps = n_steps
        self.max_seq_len = max_seq_len
        self.is_gated = is_gated
        self.is_rank0 = is_rank0
        self.prompt_len = prompt_len

        n_s, L, H, T, S = n_samples, n_layers, n_heads, n_steps, max_seq_len

        # --- All-sample accumulation arrays (CPU float32) ---
        self.mean_sum = np.zeros((n_s, L, H, S), np.float32)
        self.count    = np.zeros((n_s, 1, 1, S), np.int32)
        self.max_attn = np.zeros((n_s, L, H, S), np.float32)
        self.entropy  = np.zeros((n_s, L, H, T), np.float32)
        self.sink_sum = np.zeros((n_s, L, H, T), np.float32)

        if is_gated:
            self.eff_sum   = np.zeros((n_s, L, H, S), np.float32)
            self.gate_sum  = np.zeros((n_s, L, H), np.float32)
            self.gate_ssq  = np.zeros((n_s, L, H), np.float32)
            self.gate_cnt  = np.zeros((n_s, 1, 1), np.int32)

        # --- Sample-0 full matrix (rank 0 only) ---
        if is_rank0:
            self.matrix      = np.zeros((L, H, T, S), np.float32)
            self.sink_matrix = np.zeros((L, H, T), np.float32)
            if is_gated:
                self.gate_matrix = np.zeros((L, H, T), np.float32)

        # Per-step GPU→CPU buffers (overwritten each step)
        self.attn_buf:    dict[int, torch.Tensor] = {}   # l → [B, np, sk]
        self.entropy_buf: dict[int, torch.Tensor] = {}   # l → [B, np]
        self.sink_buf:    dict[int, torch.Tensor] = {}   # l → [B, np]
        self.gate_buf:    dict[int, torch.Tensor] = {}   # l → [1, B, np]

        self._hooks: list = []
        self._patched_modules: list = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model: torch.nn.Module) -> None:
        """Register hooks/patches on the model before the first forward call."""
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        layer_idx = 0
        for module in model.modules():
            if isinstance(module, TEDotProductAttention):
                h = module.register_forward_pre_hook(self._make_attn_hook(layer_idx))
                self._hooks.append(h)
                layer_idx += 1

        if self.is_gated:
            from megatron.core.transformer.attention import SelfAttention
            gate_idx = 0
            for module in model.modules():
                if (isinstance(module, SelfAttention)
                        and getattr(module.config, 'attention_output_gate', False)):
                    self._patch_gate(module, gate_idx)
                    gate_idx += 1

    def remove(self) -> None:
        """Remove all hooks and instance patches."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for module in self._patched_modules:
            try:
                del module._apply_output_gate
            except AttributeError:
                pass
        self._patched_modules.clear()

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_attn_hook(self, layer_idx: int):
        storage = self

        def hook(module, args):
            q, k = args[0], args[1]  # sbhd: [sq, B, np/ng, hn]
            if q.shape[0] != 1:
                return  # skip prefill (sq > 1)

            # sbhd → bshd and cast to float32 for the computation
            q = q.permute(1, 2, 0, 3).contiguous().float()  # [B, np, 1, hn]
            k = k.permute(1, 2, 0, 3).contiguous().float()  # [B, ng, sk, hn]

            B, np_, _, hn = q.shape
            ng = k.shape[1]
            if ng != np_:
                # GQA: each KV group serves np_//ng query heads (contiguous blocks, Llama convention)
                k = k.repeat_interleave(np_ // ng, dim=1)  # [B, np, sk, hn]

            scale = getattr(module, 'softmax_scale', None) or float(hn ** -0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, np, 1, sk]

            has_sink = (
                getattr(module, 'config', None) is not None
                and module.config.softmax_type in ("off-by-one", "learnable")
            )
            sk = k.shape[2]

            if has_sink:
                offset = getattr(module, 'softmax_offset', None)
                if offset is not None:
                    sink_logit = offset.detach().float().view(1, np_, 1, 1).expand(B, -1, 1, 1)
                else:
                    sink_logit = torch.zeros(B, np_, 1, 1, device=q.device, dtype=torch.float32)
                scores = torch.cat([scores, sink_logit], dim=-1)  # [B, np, 1, sk+1]

            w_full = torch.softmax(scores, dim=-1)  # [B, np, 1, sk] or [B, np, 1, sk+1]; sums to 1

            # Strip virtual sink column for positional stats
            w = w_full[..., :sk]                    # [B, np, 1, sk]
            attn_w = w.squeeze(2)                   # [B, np, sk]

            # Entropy over full distribution (sum=1 always → cross-model comparable)
            H_w = -(w_full * torch.clamp(w_full, min=1e-9).log()).sum(-1).squeeze(2)  # [B, np]

            # Sink mass: probability assigned to the virtual sink (0 for vanilla/gated)
            sink = 1.0 - attn_w.sum(-1)             # [B, np]

            storage.attn_buf[layer_idx]    = attn_w.detach().cpu()
            storage.entropy_buf[layer_idx] = H_w.detach().cpu()
            storage.sink_buf[layer_idx]    = sink.detach().cpu()

        return hook

    def _patch_gate(self, module: torch.nn.Module, layer_idx: int) -> None:
        storage = self
        original_fn = type(module)._apply_output_gate  # class-level (may be torch.compile'd)

        def patched(self_inner, x, gate):
            # gate: [sq=1, B, np, hn] — pre-sigmoid
            g = torch.sigmoid(gate.float()).mean(dim=-1)  # [1, B, np]
            storage.gate_buf[layer_idx] = g.detach().cpu()
            return original_fn(self_inner, x, gate)

        module._apply_output_gate = types.MethodType(patched, module)
        self._patched_modules.append(module)

    # ------------------------------------------------------------------
    # Per-step accumulation
    # ------------------------------------------------------------------

    def collect_step(
        self,
        t: int,
        batch_slice: slice,
        is_sample_0: bool,
    ) -> None:
        """Accumulate one decode step into the running stats.

        t: decode step index (0 .. n_steps-1)
        batch_slice: slice into n_samples dimension for the current batch
        is_sample_0: whether this batch's first sample is the global sample 0
        """
        if t == 0:
            # Verify hooks fired — catch silent failures (e.g. from torch.compile graph capture)
            missing = [l for l in range(self.n_layers) if l not in self.attn_buf]
            if missing:
                raise RuntimeError(
                    f"AttentionCapture: hooks did not fire for layers {missing} at step 0. "
                    "Check that register() was called before the first model forward."
                )

        sk = self.prompt_len + t + 1  # keys visible at this step

        for l in range(self.n_layers):
            w = self.attn_buf[l].numpy()          # [B, np, sk_actual]
            e = self.entropy_buf[l].numpy()       # [B, np]
            s = self.sink_buf[l].numpy()          # [B, np]

            self.mean_sum[batch_slice, l, :, :sk] += w
            self.count[batch_slice, 0, 0, :sk]    += 1
            np.maximum(
                self.max_attn[batch_slice, l, :, :sk],
                w,
                out=self.max_attn[batch_slice, l, :, :sk],
            )
            self.entropy[batch_slice, l, :, t]   = e
            self.sink_sum[batch_slice, l, :, t]  = s

            if self.is_gated and l in self.gate_buf:
                g = self.gate_buf[l][0, :, :].numpy()        # [B, np]
                eff_w = w * g[:, :, np.newaxis]               # [B, np, sk] per-step product
                self.eff_sum[batch_slice, l, :, :sk] += eff_w

                self.gate_sum[batch_slice, l, :]  += g
                self.gate_ssq[batch_slice, l, :] += g ** 2
                self.gate_cnt[batch_slice, 0, 0] += 1

            if is_sample_0 and self.is_rank0:
                self.matrix[l, :, t, :sk]    = w[0, :, :]    # [np, sk]
                self.sink_matrix[l, :, t]    = s[0, :]        # [np]
                if self.is_gated and l in self.gate_buf:
                    g_sample0 = self.gate_buf[l][0, 0, :].numpy()   # [np]
                    self.gate_matrix[l, :, t] = g_sample0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, out_dir: Path, rank: int) -> None:
        """Write .npz files for this rank."""
        out_dir = Path(out_dir)

        mean_attn = np.divide(
            self.mean_sum, self.count,
            out=np.zeros_like(self.mean_sum),
            where=self.count > 0,
        ).astype(np.float16)

        stats: dict[str, np.ndarray] = dict(
            mean_attn=mean_attn,
            max_attn=self.max_attn.astype(np.float16),
            entropy=self.entropy.astype(np.float16),
            sink_mass=self.sink_sum.astype(np.float16),
        )

        if self.is_gated:
            mean_eff = np.divide(
                self.eff_sum, self.count,
                out=np.zeros_like(self.eff_sum),
                where=self.count > 0,
            ).astype(np.float16)
            gate_mean = np.divide(
                self.gate_sum, self.gate_cnt,
                out=np.zeros_like(self.gate_sum),
                where=self.gate_cnt > 0,
            )
            gate_var = np.divide(
                self.gate_ssq, self.gate_cnt,
                out=np.zeros_like(self.gate_ssq),
                where=self.gate_cnt > 0,
            ) - gate_mean ** 2
            gate_std = np.sqrt(np.maximum(gate_var, 0.0))
            stats.update(
                mean_eff_attn=mean_eff,
                gate_mean=gate_mean.astype(np.float16),
                gate_std=gate_std.astype(np.float16),
            )

        np.savez(out_dir / f"attn_stats_rank{rank}.npz", **stats)

        if self.is_rank0:
            exmpl: dict[str, np.ndarray] = dict(
                matrix=self.matrix.astype(np.float16),
                sink_mass_matrix=self.sink_matrix.astype(np.float16),
                prompt_len=np.int32(self.prompt_len),
            )
            if self.is_gated:
                exmpl['gate_matrix'] = self.gate_matrix.astype(np.float16)
            np.savez(out_dir / f"attn_matrix_exmpl_rank{rank}.npz", **exmpl)
