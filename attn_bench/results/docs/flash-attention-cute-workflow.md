# flash-attn cute (FA4) workflow

`flash_attn.cute` (FlashAttention-4) is the only kernel path that supports sink attention
with real fused-kernel speed (see `hf_inference_kernel_support.md`). It's not installed by
default in `nemo_26.04_te2.15` -- the container's pip `flash_attn` predates the `cute`
submodule -- and its `main` branch currently ships a bug that breaks every forward call on
Hopper (`flash_attn/cute/utils.py::fmax()` calls `nvvm.fmax()` missing a `res` argument that
the installed `nvidia-cutlass-dsl` package's own `cute.arch.fmax` already supplies as
`T.f32()`). Same situation as PDM: an external repo we don't own, so the fix is preserved
as a patch at `attn_bench/utils/flash_attention_patch.txt` rather than a fork.

## One-time setup (per cluster user)

```bash
cd /users/$USER/scratch
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
patch -p1 --ignore-whitespace < /path/to/attn_bench/utils/flash_attention_patch.txt
```

Use `patch -p1 --ignore-whitespace`, not `git apply`, matching the PDM convention.

## Updating the patch

If upstream moves and the patch stops applying, or a new bug surfaces in another `nvvm.*`
call:

1. Edit `/users/$USER/scratch/flash-attention` directly (or a local clone + rsync).
2. Regenerate the patch from inside the flash-attention directory:
   ```bash
   git diff > attn_bench/utils/flash_attention_patch.txt
   ```
3. Commit the updated `attn_bench/utils/flash_attention_patch.txt` to this repo.

Known latent issue not yet hit or fixed: `flash_attn/cute/utils.py`'s two `nvvm.atomicrmw`
calls (dropout/backward-related, not on the plain-forward-pass path) may have the same
missing-argument problem -- not confirmed, not blocking sink inference today.

## Referencing it from a slurm script

```bash
FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention
...
pip install --quiet 'nvidia-cutlass-dsl==4.6.2'  # main's cute module floor as of 2026-08-15
python -c "
import flash_attn
flash_attn.__path__.append('$FLASH_ATTN_SRC_DIR/flash_attn')
from flash_attn.cute.interface import flash_attn_func
...
"
```

`nvidia-cutlass-dsl==4.6.2` still needs a fresh `pip install` per job (the container ships
`4.3.4`, and installs don't persist across container launches) -- only the flash-attn
source itself is worth persisting, since that's what's patched.
