"""Native pretrain_gpt entry point with injected tests.

Uses the standard model_provider (no kernel swap). Megatron args are passed
via slurm; --tests selects which test suites to inject into forward_step.

Usage:
    python3 attn_bench/tests/pretrain_gpt_native.py --tests xdoc [megatron args...]
"""
import time

_PROGRAM_START_TIME = time.time()

from functools import partial

from attn_bench.tests.util.args import add_test_args
from attn_bench.tests.util.registry import make_test_forward_step
from gpt_builders import gpt_builder
from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain, set_startup_timestamps
from megatron.training.argument_utils import gpt_config_from_args, pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider
from pretrain_gpt import forward_step as _base_forward_step
from pretrain_gpt import get_embedding_ranks, train_valid_test_datasets_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    _has_modelopt = True
except ImportError:
    _has_modelopt = False


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)
    train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    def extra_args_provider(parser):
        parser = add_test_args(parser)
        if _has_modelopt:
            parser = add_modelopt_args(parser)
        return parser

    args = parse_and_validate_args(
        extra_args_provider=extra_args_provider,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    model_cfg = gpt_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        make_test_forward_step(_base_forward_step),
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
