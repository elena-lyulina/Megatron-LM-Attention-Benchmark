"""Shared utilities for attn_bench tests (custom kernel path)."""

from functools import partial

import torch

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.utils import get_attr_wrapped_model, get_batch_on_this_cp_rank
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_batch_on_this_tp_rank, is_first_or_last_pipeline_stage

from attn_bench.kernels.attn_registry import parse_attn_kwargs, validate_attn_kwargs
from attn_bench.training.model import build_model


def kernel_model_provider(pre_process=True, post_process=True, config=None, vp_stage=None, pg_collection=None):
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)
    attn_kwargs = validate_attn_kwargs(args.attn, args.impl, parse_attn_kwargs(args.attn_kwargs))
    return build_model(
        args=args, config=config,
        attn=args.attn, impl=args.impl, attn_kwargs=attn_kwargs,
        pre_process=pre_process, post_process=post_process,
        vp_stage=vp_stage, pg_collection=pg_collection,
    )


def get_batch(data_iterator, vp_stage=None):
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None, None
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        None,  # packed_seq_params — sbhd only
    )


def make_simple_iter():
    # builds a minimal random single-batch iterator for forward_step; used by sensitivity tests
    # that just need valid input — no document structure required
    args = get_args()
    eos_id = get_tokenizer().eod
    seq_len = args.seq_length
    mbs = args.micro_batch_size
    batch = {
        'tokens': torch.randint(0, eos_id, (seq_len,)).unsqueeze(0).repeat(mbs, 1),
        'labels': torch.randint(0, eos_id, (seq_len,)).unsqueeze(0).repeat(mbs, 1),
        'loss_mask': torch.ones(mbs, seq_len),
        'position_ids': torch.arange(seq_len).unsqueeze(0).repeat(mbs, 1),
    }
    return iter([batch])


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    return loss, num_tokens, {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        tokenizer=get_tokenizer(),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=args.eod_mask_loss,
    )
    print_rank_0("building mock train, validation, and test datasets...")

    def is_dataset_built_on_rank():
        return (
            is_first_or_last_pipeline_stage(vp_stage)
            and parallel_state.get_tensor_model_parallel_rank() == 0
        )

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MockGPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, config,
    ).build()
    print_rank_0("finished creating mock GPT datasets...")
    return train_ds, valid_ds, test_ds


train_valid_test_datasets_provider.is_distributed = True
