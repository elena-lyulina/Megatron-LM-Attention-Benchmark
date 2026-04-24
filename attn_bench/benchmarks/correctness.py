"""
Correctness benchmark: track loss curves for a given (attn, impl) pair.

Trains a model for N steps with a fixed seed and logs the loss curve to w&b.
Run for different implementations of the same mechanism / different parallelism setup and
compare the resulting W&B loss curves visually.

Usage: see attn_bench/submissions/ dir
"""

from functools import partial

import wandb

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, pretrain

from attn_bench.benchmarks.args import add_benchmark_args
from attn_bench.benchmarks.common import get_batch, loss_func, model_provider, train_valid_test_datasets_provider
from attn_bench.utils.git_info import check_git_working_tree, get_git_info

# to add git info (hash, diff) to W&B after it gets initialized by Megatron
_GIT_INFO_LOGGED = False


# return_schedule_plan is needed to match the interface, not used here bc it's for the MoE
def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    # need to check inside the forward step after w&b gets initialized by megatron
    global _GIT_INFO_LOGGED
    if not _GIT_INFO_LOGGED and wandb.run is not None:
        if parallel_state.get_data_parallel_rank() == 0:
            wandb.config.update(get_git_info())
        _GIT_INFO_LOGGED = True

    vp_stage = get_attr_wrapped_model(model, "vp_stage", allow_none=True)
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params)
    return output_tensor, partial(loss_func, loss_mask)


def main():
    check_git_working_tree()

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_benchmark_args,
        args_defaults={"tokenizer_type": "NullTokenizer", "vocab_size": 32768},
    )


if __name__ == "__main__":
    main()