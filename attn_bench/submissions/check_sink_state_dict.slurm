#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:10:00
#SBATCH --job-name=check-sink-state-dict
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Comprehensive static-weight check for a converted sink checkpoint, everything in one run:
#   - every weight build_state_dict writes (Q/K/V via the real convert_qkv_weights, O-proj,
#     MLP gate/up/down, layernorms, final norm, learnable_sink) vs. the source Megatron
#     checkpoint
#   - embed_tokens.weight/lm_head.weight as stored on disk vs. what
#     AutoModelForCausalLM.from_pretrained(trust_remote_code) actually loads into memory --
#     isolates whether _tied_weights_keys=None (the fix for the earlier meta-tensor crash)
#     introduced a silent load-time bug distinct from the on-disk conversion
#   - the forward-pass wiring itself: runs the real converted model, intercepts what
#     layer 0's ALL_ATTENTION_FUNCTIONS dispatch actually receives/returns at runtime
#     (attention_mask, scaling, the real projected+roped q/k/v), and compares the kernel's
#     real output against Megatron's sink formula applied to those same real tensors --
#     isolates the dispatch boundary without re-deriving projection/rope by hand.
# The kernel itself is already validated separately by check_sink_kernel_correctness.slurm
# (both equal-heads and sink-scf1's real GQA shape) on synthetic tensors. Everything here
# passing but the full model still disagreeing with Megatron would point somewhere even
# further upstream (rope/projection correctness itself, not the dispatch or the kernel).
#
# Usage: sbatch --export=MODEL=sink-scf1 check_sink_state_dict.slurm

set -euo pipefail

CONTAINER_ENV=${CONTAINER_ENV:-nemo_26.04_te2.15}
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
# full.py's utils.py import needs PDM's src/convert on PYTHONPATH -- see full.py's docstring.
export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR/src/convert:${PYTHONPATH:-}

MODEL=${MODEL:-sink-scf1}
model_config "$MODEL"

SCRATCH_CONVERT_DIR=/iopsstor/scratch/cscs/$USER/hf-checkpoints/$EXP_NAME
MEGATRON_TORCH_CKPT=$(ls "$SCRATCH_CONVERT_DIR"/torch/iter_*/mp_rank_00/model_optim_rng.pt 2>/dev/null | head -1)
HF_DIR=$SCRATCH_CONVERT_DIR/HF

if [ -z "$MEGATRON_TORCH_CKPT" ]; then
    echo "No torch-format checkpoint found under $SCRATCH_CONVERT_DIR/torch/ -- run convert_and_validate_hf.slurm's Step 1 first."
    exit 1
fi
if [ ! -f "$HF_DIR/config.json" ]; then
    echo "$HF_DIR/config.json not found -- run convert_and_validate_hf.slurm's Step 2 first."
    exit 1
fi

SCRATCH_DIR=/users/$USER/scratch/tmp/check_sink_state_dict_$SLURM_JOB_ID
mkdir -p "$SCRATCH_DIR"

cat > "$SCRATCH_DIR/check_sink_state_dict.py" <<'PYEOF'
import sys
import torch
from safetensors.torch import load_file

def reference_sink_attention(q, k, v, sink):
    """Megatron's exact formula (fused_softmax.py: forward_torch_softmax + SoftmaxOne),
    same as check_sink_kernel_correctness.slurm. q: [B, S, Hq, D]. k/v: [B, S, Hkv, D],
    Hq a multiple of Hkv (GQA) or equal. sink: [Hq]. Returns [B, S, Hq, D]."""
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    n_rep = Hq // Hkv
    qf, kf, vf = q.float(), k.float(), v.float()
    if n_rep > 1:
        kf = kf.repeat_interleave(n_rep, dim=2)
        vf = vf.repeat_interleave(n_rep, dim=2)
    scale = D ** -0.5
    scores = torch.einsum('bshd,bthd->bhst', qf, kf) * scale
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    sink_col = sink.float().reshape(1, Hq, 1, 1).expand(B, Hq, S, 1)
    extended = torch.cat([scores, sink_col], dim=-1)
    probs = torch.softmax(extended, dim=-1)[..., :-1]
    return torch.einsum('bhst,bthd->bshd', probs, vf)

def check_attention_wiring(hf_dir):
    """Runs a real forward pass through the converted model and intercepts what layer 0's
    ALL_ATTENTION_FUNCTIONS dispatch actually receives and returns, so the kernel-vs-reference
    comparison uses the wiring's real runtime tensors instead of hand re-derived ones.

    Diagnostic-first: two prior attempts (registry patch before/after from_pretrained) both
    left `captured` empty with no exception, meaning the registered wrapper was silently never
    called. Rather than guess again, this patches the actual instantiated layer-0 self_attn's
    class-level forward (always invoked regardless of registry timing/key correctness) so we
    can directly confirm whether SinkLlamaAttention.forward runs at all and what
    config._attn_implementation resolves to at call time, before trusting the registry patch."""
    from transformers import AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype='auto', trust_remote_code=True).to('cuda')
    model.eval()

    attn_impl = model.config._attn_implementation
    print(f"model.config._attn_implementation={attn_impl!r}")
    try:
        registered_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]
        print(f"ALL_ATTENTION_FUNCTIONS[{attn_impl!r}] -> {registered_fn!r} "
              f"(module={getattr(registered_fn, '__module__', '?')})")
    except Exception as e:
        print(f"ALL_ATTENTION_FUNCTIONS[{attn_impl!r}] lookup failed: {e!r}")

    layer0_attn = model.model.layers[0].self_attn
    attn_cls = type(layer0_attn)
    print(f"layer0 self_attn class={attn_cls.__name__} (module={attn_cls.__module__})")

    called = {}
    original_class_forward = attn_cls.forward

    def wrapped_class_forward(self, **kwargs):
        called['forward'] = True
        called['learnable_sink_in_kwargs'] = 'learnable_sink' in kwargs
        called['attn_implementation_at_call'] = self.config._attn_implementation
        return original_class_forward(self, **kwargs)

    attn_cls.forward = wrapped_class_forward

    original_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]
    captured = {}

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, learnable_sink=None, **kwargs):
        result = original_fn(module, query, key, value, attention_mask, scaling,
                             dropout=dropout, learnable_sink=learnable_sink, **kwargs)
        if 'query' not in captured:  # only layer 0's (first) call
            captured['attention_mask'] = attention_mask
            captured['scaling'] = scaling
            captured['query'] = query.detach().clone()
            captured['key'] = key.detach().clone()
            captured['value'] = value.detach().clone()
            captured['learnable_sink'] = learnable_sink.detach().clone() if learnable_sink is not None else None
            captured['kernel_output'] = result[0].detach().clone()
        return result

    ALL_ATTENTION_FUNCTIONS.register(attn_impl, wrapped)

    torch.manual_seed(0)
    B, S = 1, 128
    input_ids = torch.randint(0, model.config.vocab_size, (B, S), device='cuda')
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)

    attn_cls.forward = original_class_forward
    ALL_ATTENTION_FUNCTIONS.register(attn_impl, original_fn)

    print(f"layer0 self_attn.forward called: {called.get('forward', False)}, "
          f"learnable_sink in kwargs: {called.get('learnable_sink_in_kwargs')}, "
          f"attn_implementation at call time: {called.get('attn_implementation_at_call')!r}")

    if not called.get('forward'):
        print('RESULT (attention wiring): MISMATCH (self_attn.forward never invoked)')
        return False
    if 'query' not in captured:
        print('RESULT (attention wiring): MISMATCH (forward ran but ALL_ATTENTION_FUNCTIONS dispatch was never captured -- '
              'implementation key mismatch between call time and patch time)')
        return False

    mask = captured['attention_mask']
    mask_desc = 'None' if mask is None else f'shape={tuple(mask.shape)} dtype={mask.dtype}'
    print(f"layer 0 dispatch: attention_mask={mask_desc}, scaling={captured['scaling']}, "
          f"query.shape={tuple(captured['query'].shape)} (HF [B,H,S,D] convention), "
          f"key.shape={tuple(captured['key'].shape)}, "
          f"learnable_sink.shape={tuple(captured['learnable_sink'].shape)}")

    # query/key/value arrive at the dispatch in [B, H, S, D]; reference_sink_attention wants [B, S, H, D].
    q = captured['query'].transpose(1, 2).float()
    k = captured['key'].transpose(1, 2).float()
    v = captured['value'].transpose(1, 2).float()
    ref_out = reference_sink_attention(q, k, v, captured['learnable_sink'])
    kernel_out = captured['kernel_output'].float()

    abs_diff = (kernel_out - ref_out).abs()
    atol, rtol = 1e-3, 0.02
    close = (abs_diff <= atol + rtol * ref_out.abs()).float().mean().item()
    print(f'kernel-vs-reference on real runtime q/k/v/sink (layer 0): '
          f'fraction close (atol={atol}, rtol={rtol})={close:.4f}, max_abs_diff={abs_diff.max().item():.6f}')
    ok = close >= 0.99
    print('RESULT (attention wiring):', 'CORRECT' if ok else 'MISMATCH')
    return ok

def diff_check(label, expected, actual):
    expected, actual = expected.float(), actual.float()
    if expected.shape != actual.shape:
        print(f'{label}: SHAPE MISMATCH expected={tuple(expected.shape)} actual={tuple(actual.shape)}')
        return False
    diff = (expected - actual).abs().max().item()
    ok = diff == 0
    print(f'{label}: max_abs_diff={diff:.6f} {"OK" if ok else "MISMATCH"}')
    return ok

def check_layer_weights(model_dict, hf_weights, args):
    from attn_bench.checkpoint_conversion.attn_families.full import convert_qkv_weights

    num_layers = args.num_layers
    all_match = True
    for i in range(num_layers):
        p = f'decoder.layers.{i}'
        qkv_raw = model_dict[f'{p}.self_attention.linear_qkv.weight']
        q, k, v = convert_qkv_weights(qkv_raw, args.num_attention_heads, args.num_query_groups, args.hidden_size)

        mlp_weight = model_dict[f'{p}.mlp.linear_fc1.weight']
        ffn_hidden_size = mlp_weight.shape[0] // 2

        checks = [
            (f'q_proj (layer {i})', q, hf_weights.get(f'model.layers.{i}.self_attn.q_proj.weight')),
            (f'k_proj (layer {i})', k, hf_weights.get(f'model.layers.{i}.self_attn.k_proj.weight')),
            (f'v_proj (layer {i})', v, hf_weights.get(f'model.layers.{i}.self_attn.v_proj.weight')),
            (f'o_proj (layer {i})', model_dict[f'{p}.self_attention.linear_proj.weight'],
             hf_weights.get(f'model.layers.{i}.self_attn.o_proj.weight')),
            (f'gate_proj (layer {i})', mlp_weight[:ffn_hidden_size, :],
             hf_weights.get(f'model.layers.{i}.mlp.gate_proj.weight')),
            (f'up_proj (layer {i})', mlp_weight[ffn_hidden_size:, :],
             hf_weights.get(f'model.layers.{i}.mlp.up_proj.weight')),
            (f'down_proj (layer {i})', model_dict[f'{p}.mlp.linear_fc2.weight'],
             hf_weights.get(f'model.layers.{i}.mlp.down_proj.weight')),
            (f'input_layernorm (layer {i})', model_dict[f'{p}.self_attention.linear_qkv.layer_norm_weight'],
             hf_weights.get(f'model.layers.{i}.input_layernorm.weight')),
            (f'post_attention_layernorm (layer {i})', model_dict[f'{p}.mlp.linear_fc1.layer_norm_weight'],
             hf_weights.get(f'model.layers.{i}.post_attention_layernorm.weight')),
            (f'learnable_sink (layer {i})', model_dict[f'{p}.self_attention.core_attention.softmax_offset'],
             hf_weights.get(f'model.layers.{i}.self_attn.learnable_sink')),
        ]
        for label, expected, actual in checks:
            if actual is None:
                print(f'{label}: MISSING hf key')
                all_match = False
                continue
            all_match = diff_check(label, expected, actual) and all_match

    all_match = diff_check('final norm', model_dict['decoder.final_layernorm.weight'],
                           hf_weights.get('model.norm.weight')) and all_match

    print('RESULT (layer weights):', 'ALL MATCH' if all_match else 'MISMATCH FOUND')
    return all_match

def main(megatron_ckpt_path, hf_dir):
    megatron_ckpt = torch.load(megatron_ckpt_path, weights_only=False, map_location='cpu')
    model_dict = megatron_ckpt['model']
    args = megatron_ckpt['args']
    hf_weights = load_file(f'{hf_dir}/model.safetensors')

    layer_weights_ok = check_layer_weights(model_dict, hf_weights, args)
    embed_lm_head_ok = check_embed_and_lm_head(hf_dir, hf_weights)
    wiring_ok = check_attention_wiring(hf_dir)

    if not (layer_weights_ok and embed_lm_head_ok and wiring_ok):
        sys.exit(1)

def check_embed_and_lm_head(hf_dir, hf_weights):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype='auto', trust_remote_code=True)

    checks = [
        ('model.embed_tokens.weight', model.model.embed_tokens.weight),
        ('lm_head.weight', model.lm_head.weight),
    ]
    all_match = True
    for key, loaded in checks:
        on_disk = hf_weights.get(key)
        if on_disk is None:
            print(f'{key}: not present in model.safetensors on disk')
            all_match = False
            continue
        diff = (on_disk.float() - loaded.detach().float().cpu()).abs().max().item()
        ok = diff == 0
        print(f'{key}: max_abs_diff(on-disk vs from_pretrained-loaded)={diff:.6f} {"OK" if ok else "MISMATCH"}')
        all_match = all_match and ok

    print('RESULT (embed/lm_head):', 'ALL MATCH' if all_match else 'MISMATCH FOUND')
    return all_match

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
PYEOF

srun --environment=$CONTAINER_ENV bash -c "
export PYTHONPATH='${PYTHONPATH}'
export FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention
pip install --quiet 'nvidia-cutlass-dsl==4.6.2'
python '$SCRATCH_DIR/check_sink_state_dict.py' '$MEGATRON_TORCH_CKPT' '$HF_DIR'
"

rm -rf "$SCRATCH_DIR"
