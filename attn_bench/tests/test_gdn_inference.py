# GDN inference tests: verify GDN generation is correct.
# M0 (this file): the cacheless quadratic *oracle* generator + its self-check (oracle output ==
# teacher forcing over the committed sequence). No core change -- runs against today's code.
# M1 (later): add the cached/incremental decode vs oracle equivalence assertion, once
# GatedDeltaNet supports inference. See attn_bench/_plans/gdn_inference_plan.md.

import torch

from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.training import get_tokenizer, print_rank_0


@torch.no_grad()
def greedy_cacheless(model, prompt_ids, suffix_length):
    # Quadratic oracle: re-run the full forward over prompt+generated each step, no inference_context.
    # Same path as the training forward (chunked kernel), so it is the ground-truth generator.
    # prompt_ids: [B, P]  ->  returns generated suffix [B, suffix_length]
    seq = prompt_ids
    generated = []
    for _ in range(suffix_length):
        B, L = seq.shape
        pos = torch.arange(L, dtype=torch.long, device=seq.device).unsqueeze(0).expand(B, -1)
        logits = model(seq, pos, attention_mask=None)            # [B, L, V]; logits[:, i] -> token i+1
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        generated.append(next_tok)
        seq = torch.cat([seq, next_tok], dim=1)
    return torch.cat(generated, dim=1)


@torch.no_grad()
def greedy_cached(model, prompt_ids, suffix_length):
    # The path under test: prefill the prompt into a StaticInferenceContext (writes the GDN conv +
    # recurrent state into the cache), then decode one token at a time. Mirrors greedy_generate in
    # attn_bench/evaluation/megatron_inference_sparse.py, kept local so the test pulls no eval deps.
    from megatron.core.inference.contexts import StaticInferenceContext

    B, P = prompt_ids.shape
    device = prompt_ids.device
    ctx = StaticInferenceContext(max_batch_size=B, max_sequence_length=P + suffix_length)
    ctx.reset()
    ctx.enable_prefill_mode()

    pos = torch.arange(P, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
    logits = model(prompt_ids, pos, attention_mask=None, inference_context=ctx, runtime_gather_output=True)
    ctx.sequence_len_offset = P
    ctx.enable_decode_mode()

    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
    generated = [next_tok]
    for _ in range(suffix_length - 1):
        pos = torch.full((B, 1), ctx.sequence_len_offset, dtype=torch.long, device=device)
        logits = model(next_tok, pos, attention_mask=None, inference_context=ctx, runtime_gather_output=True)
        ctx.sequence_len_offset += 1
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok)
    return torch.cat(generated, dim=1)


def _tiny_prompt(model, batch, prefix_length):
    # a [batch, prefix_length] prompt of valid token ids, on the model's device
    device = next(model.parameters()).device
    eos = get_tokenizer().eod
    return torch.randint(0, eos, (batch, prefix_length), dtype=torch.long, device=device)


# tiny configs (batch, prefix_length, suffix_length): small enough to eyeball + cheap, B>1, no offset
# axis (the model is --position-embedding-type none, so absolute offset cannot change the computation)
_ORACLE_CONFIGS = [(2, 4, 8), (2, 8, 16), (3, 16, 32)]


def _make_test_oracle_selfcheck(base_forward_step):
    # base_forward_step is unused (the oracle calls the model directly for logits), but kept in the
    # factory signature so registry.py wires this suite the same way as the others.
    def test_oracle_selfcheck(model):
        print_rank_0("\n### Test: oracle_selfcheck ###")
        if not any(isinstance(m, GatedDeltaNet) for _, m in model.named_modules()):
            print_rank_0("[FAIL] oracle_selfcheck: no GatedDeltaNet module found in model")
            return False

        was_training = model.training
        model.eval()
        torch.manual_seed(1234)

        ok = True
        for B, P, S in _ORACLE_CONFIGS:
            prompt = _tiny_prompt(model, B, P)
            gen = greedy_cacheless(model, prompt, S)             # [B, S]

            # One teacher-forced forward over the full sequence must reproduce the generated suffix:
            # the model is causal, so the last-position logit over a prefix equals the same-position
            # logit over the longer committed sequence. This validates the oracle is self-consistent.
            full = torch.cat([prompt, gen], dim=1)               # [B, P+S]
            pos = torch.arange(P + S - 1, dtype=torch.long, device=full.device).unsqueeze(0).expand(B, -1)
            logits = model(full[:, :-1], pos, attention_mask=None)   # [B, P+S-1, V]
            pred = logits[:, P - 1:, :].argmax(dim=-1)           # [B, S] predictions for suffix positions
            match = torch.equal(pred, gen)

            print_rank_0(f"  cfg B={B} P={P} S={S}: teacher-forcing match={match}  gen[0]={gen[0].tolist()}")
            if not match:
                ok = False

        if was_training:
            model.train()

        print_rank_0(f"[{'PASS' if ok else 'FAIL'}] oracle_selfcheck: oracle == teacher forcing")
        return ok

    return test_oracle_selfcheck


def _make_test_decode_matches_oracle(base_forward_step):
    # the M1 gate: the cached/incremental decode must reproduce the quadratic oracle token for token.
    def test_decode_matches_oracle(model):
        print_rank_0("\n### Test: decode_matches_oracle ###")
        if not any(isinstance(m, GatedDeltaNet) for _, m in model.named_modules()):
            print_rank_0("[FAIL] decode_matches_oracle: no GatedDeltaNet module found in model")
            return False

        was_training = model.training
        model.eval()
        torch.manual_seed(1234)

        ok = True
        for B, P, S in _ORACLE_CONFIGS:
            prompt = _tiny_prompt(model, B, P)
            ref = greedy_cacheless(model, prompt, S)             # quadratic oracle (ground truth)
            got = greedy_cached(model, prompt, S)                # cached path under test
            match = torch.equal(ref, got)

            print_rank_0(f"  cfg B={B} P={P} S={S}: cached==oracle match={match}")
            print_rank_0(f"    oracle[0]={ref[0].tolist()}")
            print_rank_0(f"    cached[0]={got[0].tolist()}")
            if not match:
                ok = False

        if was_training:
            model.train()

        print_rank_0(f"[{'PASS' if ok else 'FAIL'}] decode_matches_oracle: cached decode == quadratic oracle")
        return ok

    return test_decode_matches_oracle


def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'gdn_inference' suite;
    # each returned function has signature (model)->bool
    return [
        _make_test_oracle_selfcheck(base_forward_step),
        _make_test_decode_matches_oracle(base_forward_step),
    ]
