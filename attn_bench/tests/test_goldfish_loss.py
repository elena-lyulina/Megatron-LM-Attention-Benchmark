# Goldfish loss tests (--tests goldfish=...). Both driven by the run's real flags:
#   test_goldfish_masking -- the dataset produces the right loss_mask zeros. Reads goldfish
#       settings from the real config (core_gpt_dataset_config_from_args) and runs the real
#       GPTDataset.__getitem__ on MockGPTDatasets. No model forward.
#   test_goldfish_loss    -- those zeros actually reduce the loss. Runs the real forward_step
#       on a micro-batch of mbs DISTINCT dataset samples (each with its own goldfish loss_mask)
#       vs the same tokens without goldfish, and checks the goldfish batch counts fewer tokens
#       and has smaller loss.

import torch

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import \
    BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import (_GOLDFISH_TOKEN_ID,
                                                GPTDatasetConfig,
                                                MockGPTDataset,
                                                _create_hash_table,
                                                apply_goldfish)
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.utils import is_first_or_last_pipeline_stage

_RATE_TOL = 0.5    # ~1/k drop rate: loose multiplicative tolerance, pooled over samples
_NUM_SAMPLES = 32  # pooled samples for a stable rate estimate


def _real_goldfish_from_args():
    # Build the config the SAME way training does, so the slurm --goldfish-* flags flow
    # args -> core_gpt_dataset_config_from_args -> data_args -> config. Reading these off the
    # real config is what tests the pretrain_gpt.py data_args wiring. Imported lazily to avoid
    # the entry-point sys.path issue.
    from pretrain_gpt import core_gpt_dataset_config_from_args
    cfg = core_gpt_dataset_config_from_args(get_args())
    return cfg.goldfish_loss, cfg.goldfish_k, cfg.goldfish_h


def _build_mock_ds(goldfish, k, h, seq_len, num_samples, vp_stage=None):
    # Mirrors attn_bench/tests/util/common.py's provider, adding the goldfish fields and
    # forcing eod_mask_loss=False so the ONLY loss_mask zeros are goldfish drops.
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=seq_len,
        tokenizer=get_tokenizer(),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        goldfish_loss=goldfish,
        goldfish_k=k,
        goldfish_h=h,
    )

    def is_dataset_built_on_rank():
        return (
            is_first_or_last_pipeline_stage(vp_stage)
            and parallel_state.get_tensor_model_parallel_rank() == 0
        )

    train_ds, _, _ = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [num_samples, num_samples, num_samples], is_dataset_built_on_rank, config,
    ).build()
    return train_ds


def _expected_drops(labels, k, h):
    # Independently recompute which positions goldfish drops, using the same functions the
    # dataset uses; returns a boolean mask over positions.
    table = _create_hash_table(device=labels.device)
    gf = apply_goldfish(labels, _GOLDFISH_TOKEN_ID, k, table, h)
    return gf == _GOLDFISH_TOKEN_ID


def _batch(samples):
    # Stack a list of per-sample dicts into one micro-batch ([mbs, seq_len] tensors), exactly as
    # the dataloader collate does.
    return {
        key: torch.stack([s[key] for s in samples])
        for key in ("tokens", "labels", "loss_mask", "position_ids")
    }


def test_goldfish_masking(model):
    print_rank_0("\n### Test: goldfish_masking ###")
    args = get_args()
    seq_len = args.seq_length
    mbs = args.micro_batch_size
    gf, k, h = _real_goldfish_from_args()   # from the real args->config path, not hand-built
    print_rank_0(f"  config goldfish_loss={gf}  k={k}  h={h}  seq_len={seq_len}  mbs={mbs}")

    # Goldfish is applied per-sample in __getitem__; batching must keep each row's own mask.
    # Check on real micro-batches of mbs distinct samples; several batches for a stable rate.
    num_batches = max(1, _NUM_SAMPLES // mbs)
    on_ds = _build_mock_ds(gf, k, h, seq_len, num_batches * mbs)      # run's real goldfish settings
    off_ds = _build_mock_ds(False, k, h, seq_len, num_batches * mbs)  # control

    errors = []
    total_tokens = total_drops = 0
    for b in range(num_batches):
        rows = range(b * mbs, (b + 1) * mbs)
        on = _batch([on_ds[i] for i in rows])
        off = _batch([off_ds[i] for i in rows])
        for r in range(mbs):
            labels = on["labels"][r]
            on_zeros = (on["loss_mask"][r] == 0.0)
            expected = _expected_drops(labels, k, h)
            # each row of the batch must carry its OWN goldfish mask (apply_goldfish wired in)
            if not torch.equal(on_zeros, expected):
                errors.append(f"  batch {b} row {r}: loss_mask zeros != apply_goldfish drops")
            if on_zeros[: h - 1].any():
                errors.append(f"  batch {b} row {r}: a drop in the first h-1={h - 1} positions")
            if (off["loss_mask"][r] == 0.0).any():
                errors.append(f"  batch {b} row {r}: goldfish OFF but loss_mask has zeros")
            total_tokens += labels.numel()
            total_drops += int(on_zeros.sum())

    # CRITICAL wiring assertion: the slurm flag actually reached the config and masked tokens.
    # If the args->config->data_args passthrough were broken, gf would be False -> zero drops.
    if total_drops == 0:
        errors.append("  no tokens dropped -- goldfish did not reach the config (check --goldfish-loss wiring)")

    #  checks that fraction of drops is as expected (1/k)
    rate = total_drops / max(total_tokens, 1)
    target = 1.0 / k
    if total_drops and not (target * (1 - _RATE_TOL) <= rate <= target * (1 + _RATE_TOL)):
        errors.append(f"  drop rate {rate:.4f} not within +-{_RATE_TOL:.0%} of 1/k={target:.4f}")
    print_rank_0(f"  batches={num_batches} mbs={mbs}  drops={total_drops}/{total_tokens}  rate={rate:.4f}  target 1/k={target:.4f}")

    if errors:
        print_rank_0(f"\n[FAIL] goldfish_masking  ({len(errors)} errors):")
        for e in errors[:10]:
            print_rank_0(e)
        return False
    print_rank_0("\n[PASS] goldfish_masking: per-row loss_mask==drops across mbs batches; off=no masking; rate~1/k; first h-1 safe")
    return True


def _iter_batch(samples):
    # A micro-batch of distinct samples (one row each) wrapped as the one-step iterator
    # base_forward_step / get_batch expect. Each row keeps its own goldfish loss_mask.
    return iter([_batch(samples)])


def _make_test_goldfish_loss(base_forward_step):
    # Returns test_goldfish_loss with base_forward_step captured in a closure (same pattern as
    # test_xdoc_attention: the entry point owns base_forward_step and passes it into register).
    def test_goldfish_loss(model):
        print_rank_0("\n### Test: goldfish_loss ###")
        args = get_args()
        gf, k, h = _real_goldfish_from_args()
        mbs = args.micro_batch_size

        # A real micro-batch of mbs DISTINCT samples, each with its own goldfish loss_mask from
        # __getitem__ (this is what tests mbs > 1). off = same tokens, no goldfish (full baseline);
        # goldfish only changes loss_mask, not tokens, so same seed => same tokens per row.
        on_ds = _build_mock_ds(gf, k, h, args.seq_length, mbs)
        off_ds = _build_mock_ds(False, k, h, args.seq_length, mbs)
        goldfish_rows = [on_ds[i] for i in range(mbs)]
        full_rows = [off_ds[i] for i in range(mbs)]

        for i in range(mbs):
            if not torch.equal(goldfish_rows[i]["tokens"], full_rows[i]["tokens"]):
                print_rank_0(f"[FAIL] goldfish_loss: sample {i} tokens differ; cannot isolate the loss effect")
                return False

        drops = int(sum(
            full_rows[i]["loss_mask"].sum() - goldfish_rows[i]["loss_mask"].sum()
            for i in range(mbs)
        ))

        was_training = model.training
        model.eval()
        with torch.no_grad():
            out_full, loss_fn_full = base_forward_step(_iter_batch(full_rows), model)
            loss_full, num_tokens_full, _ = loss_fn_full(out_full)
            out_goldfish, loss_fn_goldfish = base_forward_step(_iter_batch(goldfish_rows), model)
            loss_goldfish, num_tokens_goldfish, _ = loss_fn_goldfish(out_goldfish)
        if was_training:
            model.train()

        num_tokens_diff = int(num_tokens_full - num_tokens_goldfish)
        loss_diff = float(loss_full - loss_goldfish)
        print_rank_0(f"  goldfish drops={drops}")
        print_rank_0(
            f"  num_tokens: full={int(num_tokens_full)} goldfish={int(num_tokens_goldfish)} "
            f"diff={num_tokens_diff}"
        )
        print_rank_0(
            f"  loss: full={float(loss_full):.4f} goldfish={float(loss_goldfish):.4f} diff={loss_diff:.4f}"
        )

        # If loss_mask were ignored downstream, both diffs would be 0. A used mask excludes
        # exactly the dropped tokens from the count and removes their (positive) loss.
        errors = []
        if drops == 0:
            errors.append("  no goldfish drops in the sample -- cannot check the loss effect")
        if num_tokens_diff != drops:
            errors.append(f"  num_tokens dropped {num_tokens_diff}, expected {drops} (loss_mask not honored?)")
        if loss_diff <= 0:
            errors.append("  goldfish loss not smaller than full loss (dropped tokens still counted?)")

        if errors:
            print_rank_0("\n[FAIL] goldfish_loss:")
            for e in errors:
                print_rank_0(e)
            return False
        print_rank_0(f"\n[PASS] goldfish_loss: dataset's goldfish loss_mask excludes dropped tokens from loss (count -{drops}, loss -{loss_diff:.4f})")
        return True

    return test_goldfish_loss


def register(base_forward_step):
    # 'goldfish' suite:
    #   test_goldfish_masking -- the dataset masks the right tokens (no forward).
    #   test_goldfish_loss    -- those masked tokens are excluded from the loss (real forward).
    return [test_goldfish_masking, _make_test_goldfish_loss(base_forward_step)]
