"""
Attention capture for memorization analysis.

Captures full causal attention maps (prefill + decode) during greedy generation and
averages them into Rouge-L buckets, so we can compare attention flow between memorized
and non-memorized samples. Nothing is collapsed at capture time: from the full map you
recover BOS attention (column 0), virtual-sink mass (1 - row sum, for sink/obo models),
entropy (row-wise H), etc. yourself, at any granularity.

Registers a forward pre-hook on every TEDotProductAttention module (pure observation,
recomputing the softmax from q/k/v; model output unchanged) and, for the gated model,
patches _apply_output_gate to read the per-position output gate.

Three families of files are written per rank (aggregated across ALL repetition buckets):
  - attn_scores_rouge_l_{00-01..09-10}_rank{N}.npz
        mean attention map [L, H, S, S] per Rouge-L bucket (+ sample count).
        S = prompt_len + suffix_length - 1; row i = query position i, col j = key position j.
        Rows are NOT renormalized: for sink/off-by-one models the row-sum deficit is the
        virtual-sink mass.
  - norm_attn_rouge_l_{00-01..09-10}_rank{N}.npz
        mean norm-based map [L, H, S, S] per bucket (Kobayashi et al., "Attention is Not
        Only a Weight"): n(i,j,h) = alpha_{ij,h} * || v_{j,h} (*) g_{i,h} ||_2, where g is
        the sigmoid output gate at the query position (g == 1 for non-gated models). The
        output projection W_O is intentionally not folded in.
  - gating_scores_rank{N}.npz   (gated model only)
        per-(Rouge-L bucket, layer, head) histogram of sigmoid(gate) over all gate
        elements / query positions / samples, for normalized-density plots.

Usage:
    capture = AttentionCapture(n_layers, n_heads, prompt_len, suffix_length, is_gated)
    capture.register(model)
    ...
    for batch in loader:
        capture.begin_batch(B)
        generated = greedy_generate(model, prompt, suffix_length,
                                    prefill_callback=capture.collect_prefill,
                                    step_callback=capture.collect_decode)
        ...                                    # compute rouge_l per sample
        capture.flush_batch(rouge_l)           # route this batch into Rouge-L buckets
    capture.save(out_dir, rank)
    capture.remove()
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch

N_BUCKETS = 10        # Rouge-L buckets: [0.0,0.1), [0.1,0.2), ..., [0.9,1.0]
N_GATE_BINS = 100     # gating-score histogram resolution over [0, 1]


def bucket_label(bi: int) -> str:
    """'00-01', '01-02', ..., '09-10' for bucket index 0..9."""
    return f"{bi:02d}-{bi + 1:02d}"


class AttentionCapture:
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        prompt_len: int,
        suffix_length: int,
        is_gated: bool,
        n_buckets: int = N_BUCKETS,
        n_gate_bins: int = N_GATE_BINS,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.prompt_len = prompt_len
        self.suffix_length = suffix_length
        self.is_gated = is_gated
        self.n_buckets = n_buckets
        self.n_gate_bins = n_gate_bins

        # Full causal map is square: query positions 0..S-1 (the last sequence position is
        # never a query), key positions 0..S-1.
        self.seq_len = prompt_len + suffix_length - 1

        L, H, S = n_layers, n_heads, self.seq_len

        # --- Bucket accumulators (float32 sums + per-bucket sample counts) ---
        self.attn_sum = np.zeros((n_buckets, L, H, S, S), np.float32)
        self.norm_sum = np.zeros((n_buckets, L, H, S, S), np.float32)
        self.count    = np.zeros((n_buckets,), np.int64)
        if is_gated:
            self.gate_hist = np.zeros((n_buckets, L, H, n_gate_bins), np.int64)

        # --- Per-batch buffers (allocated in begin_batch) ---
        self.batch_attn: np.ndarray | None = None       # [B, L, H, S, S] f16
        self.batch_norm: np.ndarray | None = None        # [B, L, H, S, S] f16
        self.batch_gate_hist: np.ndarray | None = None   # [B, L, H, n_gate_bins] i64
        self._batch_size = 0

        # --- Per-forward raw buffers (overwritten each forward) ---
        self.w_buf:    dict[int, np.ndarray]    = {}   # l -> [B, np, Lq, Lk] f32 (CPU, sink stripped)
        self.v_buf:    dict[int, torch.Tensor]  = {}   # l -> [B, np, Lk, hn] f32 (GPU)
        self.gate_buf: dict[int, torch.Tensor]  = {}   # l -> [Lq, B, np, hn] pre-sigmoid (GPU)

        self._hooks: list = []
        self._patched_modules: list = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model: torch.nn.Module) -> None:
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
            q, k, v = args[0], args[1], args[2]  # sbhd: [s, B, np/ng, hn]

            # sbhd -> bshd, float32 for the recomputed softmax
            q = q.permute(1, 2, 0, 3).contiguous().float()  # [B, np, Lq, hn]
            k = k.permute(1, 2, 0, 3).contiguous().float()  # [B, ng, Lk, hn]
            v = v.permute(1, 2, 0, 3).contiguous().float()  # [B, ng, Lk, hn]

            B, np_, Lq, hn = q.shape
            ng = k.shape[1]
            if ng != np_:
                # GQA: each KV group serves np_//ng query heads (contiguous, Llama convention)
                rep = np_ // ng
                k = k.repeat_interleave(rep, dim=1)  # [B, np, Lk, hn]
                v = v.repeat_interleave(rep, dim=1)  # [B, np, Lk, hn]
            Lk = k.shape[2]

            scale = getattr(module, 'softmax_scale', None) or float(hn ** -0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, np, Lq, Lk]

            # Causal mask only matters when we process >1 query at once (prefill).
            if Lq > 1:
                causal = torch.triu(
                    torch.ones(Lq, Lk, dtype=torch.bool, device=scores.device), diagonal=1
                )
                scores = scores.masked_fill(causal, float('-inf'))

            has_sink = (
                getattr(module, 'config', None) is not None
                and module.config.softmax_type in ("off-by-one", "learnable")
            )
            if has_sink:
                offset = getattr(module, 'softmax_offset', None)
                if offset is not None:
                    sink_logit = offset.detach().float().view(1, np_, 1, 1).expand(B, -1, Lq, 1)
                else:
                    sink_logit = torch.zeros(B, np_, Lq, 1, device=q.device, dtype=torch.float32)
                scores = torch.cat([scores, sink_logit], dim=-1)  # [B, np, Lq, Lk+1]

            w_full = torch.softmax(scores, dim=-1)
            w = w_full[..., :Lk]  # strip virtual-sink column; rows may sum < 1 for sink/obo

            # Big tensor -> CPU immediately (prefill across all layers would not fit on GPU).
            storage.w_buf[layer_idx] = w.cpu().numpy()
            storage.v_buf[layer_idx] = v  # small, kept on GPU for the norm matmul

        return hook

    def _patch_gate(self, module: torch.nn.Module, layer_idx: int) -> None:
        storage = self
        original_fn = type(module)._apply_output_gate  # class-level (may be torch.compile'd)

        def patched(self_inner, x, gate):
            # gate: [Lq, B, np, hn] -- pre-sigmoid
            storage.gate_buf[layer_idx] = gate.detach().float()
            return original_fn(self_inner, x, gate)

        module._apply_output_gate = types.MethodType(patched, module)
        self._patched_modules.append(module)

    # ------------------------------------------------------------------
    # Per-batch lifecycle
    # ------------------------------------------------------------------

    def begin_batch(self, batch_size: int) -> None:
        """Allocate fresh per-sample buffers for a new batch."""
        L, H, S = self.n_layers, self.n_heads, self.seq_len
        self._batch_size = batch_size
        self.batch_attn = np.zeros((batch_size, L, H, S, S), np.float16)
        self.batch_norm = np.zeros((batch_size, L, H, S, S), np.float16)
        if self.is_gated:
            self.batch_gate_hist = np.zeros((batch_size, L, H, self.n_gate_bins), np.int64)
        self.w_buf.clear()
        self.v_buf.clear()
        self.gate_buf.clear()

    def collect_prefill(self) -> None:
        """Fill query rows 0..prompt_len-1 from the prefill forward."""
        missing = [l for l in range(self.n_layers) if l not in self.w_buf]
        if missing:
            raise RuntimeError(
                f"AttentionCapture: attention hooks did not fire for layers {missing} during "
                "prefill. Check that register() ran before the first model forward."
            )
        self._collect(row_start=0)

    def collect_decode(self, t: int) -> None:
        """Fill the single query row for decode step t (query position prompt_len + t)."""
        self._collect(row_start=self.prompt_len + t)

    def _collect(self, row_start: int) -> None:
        for l in range(self.n_layers):
            w = self.w_buf[l]                 # [B, np, Lq, Lk] f32 (CPU)
            v = self.v_buf[l]                 # [B, np, Lk, hn] f32 (GPU)
            B, np_, Lq, Lk = w.shape

            g_sig = None
            if self.is_gated and l in self.gate_buf:
                # [Lq, B, np, hn] -> [B, np, Lq, hn]
                g_sig = torch.sigmoid(self.gate_buf[l]).permute(1, 2, 0, 3).contiguous()

            # --- norm-based map: alpha * || v_j (*) g_i || ---
            if g_sig is None:
                vnorm = v.norm(dim=-1)                              # [B, np, Lk]
                vnorm = vnorm.cpu().numpy()[:, :, None, :]           # [B, np, 1, Lk]
                norm_attn = w * vnorm                                # broadcast over query rows
            else:
                g2 = (g_sig * g_sig)                                 # [B, np, Lq, hn]
                v2 = (v * v)                                          # [B, np, Lk, hn]
                vnormM = torch.sqrt(torch.matmul(g2, v2.transpose(-2, -1)).clamp_min(0))  # [B,np,Lq,Lk]
                norm_attn = w * vnormM.cpu().numpy()

            # --- write into per-sample buffers ---
            r0 = row_start
            r1 = row_start + Lq
            self.batch_attn[:, l, :, r0:r1, :Lk] = w.astype(np.float16)
            self.batch_norm[:, l, :, r0:r1, :Lk] = norm_attn.astype(np.float16)

            # --- gating histogram over this forward's gate elements ---
            if g_sig is not None:
                self._accumulate_gate_hist(l, g_sig)

    def _accumulate_gate_hist(self, layer_idx: int, g_sig: torch.Tensor) -> None:
        # g_sig: [B, np, Lq, hn] in [0, 1]
        gi = g_sig.cpu().numpy()
        bidx = np.clip((gi * self.n_gate_bins).astype(np.int64), 0, self.n_gate_bins - 1)
        Bn, Hn, Lq, hn = bidx.shape
        b_coord = np.broadcast_to(np.arange(Bn)[:, None, None, None], bidx.shape)
        h_coord = np.broadcast_to(np.arange(Hn)[None, :, None, None], bidx.shape)
        np.add.at(
            self.batch_gate_hist[:, layer_idx, :, :],
            (b_coord.ravel(), h_coord.ravel(), bidx.ravel()),
            1,
        )

    def flush_batch(self, rouge_l) -> None:
        """Route each sample's accumulated maps into its Rouge-L bucket."""
        rouge_l = np.asarray(rouge_l, dtype=np.float64)
        for b in range(self._batch_size):
            bk = int(np.clip(int(rouge_l[b] * self.n_buckets), 0, self.n_buckets - 1))
            self.attn_sum[bk] += self.batch_attn[b]
            self.norm_sum[bk] += self.batch_norm[b]
            self.count[bk] += 1
            if self.is_gated:
                self.gate_hist[bk] += self.batch_gate_hist[b]

        self.batch_attn = self.batch_norm = self.batch_gate_hist = None
        self.w_buf.clear()
        self.v_buf.clear()
        self.gate_buf.clear()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, out_dir: Path, rank: int) -> None:
        """Write per-bucket attention / norm maps and the gating histogram for this rank."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for bi in range(self.n_buckets):
            n = int(self.count[bi])
            label = bucket_label(bi)
            if n > 0:
                attn_mean = (self.attn_sum[bi] / n).astype(np.float16)
                norm_mean = (self.norm_sum[bi] / n).astype(np.float16)
            else:
                attn_mean = self.attn_sum[bi].astype(np.float16)
                norm_mean = self.norm_sum[bi].astype(np.float16)

            np.savez(
                out_dir / f"attn_scores_rouge_l_{label}_rank{rank}.npz",
                mean=attn_mean, count=np.int64(n), prompt_len=np.int32(self.prompt_len),
            )
            np.savez(
                out_dir / f"norm_attn_rouge_l_{label}_rank{rank}.npz",
                mean=norm_mean, count=np.int64(n), prompt_len=np.int32(self.prompt_len),
            )

        if self.is_gated:
            edges = np.linspace(0.0, 1.0, self.n_gate_bins + 1, dtype=np.float32)
            np.savez(
                out_dir / f"gating_scores_rank{rank}.npz",
                hist=self.gate_hist, bin_edges=edges, count=self.count,
            )