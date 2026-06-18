# PDM Workflow

PDM is an external repo not owned by us, so we cannot push changes directly.
Instead, we preserve a patch at `attn_bench/utils/PDM_patch.txt` so it can be
applied to any PDM clone.

## Making changes

1. Edit PDM locally.
2. Rsync to the cluster to propagate changes.
3. Regenerate the patch from inside the PDM directory:
   ```bash
   git diff > attn_bench/utils/PDM_patch.txt
   ```
4. Commit the updated `attn_bench/utils/PDM_patch.txt` to this repo.

## Applying the patch (fresh PDM clone)

From inside the PDM directory:
```bash
patch -p1 --ignore-whitespace < /path/to/attn_bench/utils/PDM_patch.txt
```

Use `patch -p1 --ignore-whitespace`, **not** `git apply`. The PDM repo has trailing
whitespace on some blank lines that `git apply --ignore-whitespace` does not handle in
context lines, but `patch` does.