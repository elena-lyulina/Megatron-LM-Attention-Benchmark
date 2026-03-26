"""
Correctness benchmark: track loss curves for a given (attn, impl) pair.

Trains a model for N steps with a fixed seed and logs the loss curve to w&b.
Run for different implementations of the same mechanism / different parallelism setup and
compare the resulting W&B loss curves visually.

Usage: see attn_bench/submissions/ dir
"""

from functools import partial

import torch
import wandb

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.utils import get_batch_on_this_cp_rank, get_attr_wrapped_model
from megatron.training import get_args, pretrain, print_rank_0, get_tokenizer
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import is_first_or_last_pipeline_stage, get_batch_on_this_tp_rank

from attn_bench.benchmarks.args import add_benchmark_args
from attn_bench.training.model import build_model
from attn_bench.utils.git_info import check_git_working_tree, get_git_info

# to add git info (hash, diff) to W&B after it gets initialized by Megatron
_GIT_INFO_LOGGED = False


#  Parse args and build the model with the custom attention mechanism
def model_provider(pre_process=True, post_process=True, config=None, vp_stage=None, pg_collection=None):
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)
    return build_model(
        args=args,
        config=config,
        attn=args.attn,
        impl=args.impl,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

# Simplified for benchmarks
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
        None,  # packed_seq_params — sbhd only (yet)
    )

# Also simplified
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    return loss, num_tokens, {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}


# return_schedule_plan is needed to match the interface, not used here bc it's for the MoE
def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    # need to check inside the forward step after w&b gets initialized by megatron
    global _GIT_INFO_LOGGED
    if not _GIT_INFO_LOGGED and wandb.run is not None:
        if parallel_state.get_data_parallel_rank() == 0:
            wandb.config.update(get_git_info())
        _GIT_INFO_LOGGED = True

    vp_stage = get_attr_wrapped_model(model, "vp_stage", allow_none=True)
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
        data_iterator, vp_stage
    )
    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels,
        loss_mask=loss_mask, packed_seq_params=packed_seq_params,
    )
    return output_tensor, partial(loss_func, loss_mask)



def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    # Uses MockGPTDataset which is enough for comparing loss curves
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
        MockGPTDataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config,
    ).build()
    print_rank_0("finished creating mock GPT datasets...")
    return train_ds, valid_ds, test_ds


train_valid_test_datasets_provider.is_distributed = True


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
