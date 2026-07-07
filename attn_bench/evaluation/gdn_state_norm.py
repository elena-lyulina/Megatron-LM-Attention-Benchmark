"""Log the Frobenius norm of each GDN recurrent state along a sequence.

The chunk kernel only returns the state at the end of the span it is given. To read it
along the way we feed the kernel fixed-size slices and carry the state across them (same
result as one call, just observable at each boundary). Installed by swapping each
GatedDeltaNet's gated_delta_rule; no megatron/core change.

Norms are kept per layer, per boundary (every state_chunk tokens), per head, averaged
over a bucket's samples via sum / sqsum / count. A final partial tail is still run (so
the output stays complete) but not logged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from megatron.core.ssm.gated_delta_net import GatedDeltaNet


class StateNormAccumulator:
    """Accumulates per-layer, per-boundary, per-head state norms over a bucket.

    Shared by all GatedDeltaNet wrappers. Call order: reset_bucket once, then per sequence
    reset_sequence -> forward -> accumulate, then reduce + save.
    """

    def __init__(self, state_chunk, layer_ids, num_heads, device):
        self.state_chunk = state_chunk
        self.layer_ids = sorted(layer_ids)  # fixed order -> row index in the saved array
        self._row = {lid: i for i, lid in enumerate(self.layer_ids)}
        self.num_heads = num_heads
        self.device = device

        # per-sequence scratch: layer_number -> [num_full_boundaries, num_heads]
        self._seq = {}
        # bucket accumulators, allocated in begin_bucket (same shape on every rank)
        self._max_b = 0
        self._sum = None    # [num_layers, max_b, num_heads] float64
        self._sqsum = None
        self._count = None  # [max_b] float64

    ### per bucket ###

    def reset_bucket(self, max_tokens):
        # max_tokens = longest kernel input length (S1) in this bucket. A boundary lands
        # every state_chunk tokens, so the most boundaries any sequence can have is
        # max_tokens // state_chunk. Allocate deterministically (not lazily) so every rank
        # holds the same shape for the all_reduce below.
        self._max_b = max(1, max_tokens // self.state_chunk)
        shape = (len(self.layer_ids), self._max_b, self.num_heads)
        self._sum = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._sqsum = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._count = torch.zeros(self._max_b, dtype=torch.float64, device=self.device)

    ### per sequence ###

    def reset_sequence(self):
        self._seq = {}

    def record(self, layer_number, norms):
        # norms: [num_full_boundaries, batch, num_heads]; batch is 1 in this eval.
        self._seq[layer_number] = norms[:, 0, :]

    def accumulate(self):
        # Fold this sequence's per-layer boundary norms into the bucket totals. All layers
        # see the same sequence length, so they share one boundary count (incremented once).
        if not self._seq:
            return
        n_full = min(t.shape[0] for t in self._seq.values())
        if n_full == 0:
            return
        for lid, norms in self._seq.items():
            row = self._row[lid]
            v = norms[:n_full].double()
            self._sum[row, :n_full] += v
            self._sqsum[row, :n_full] += v * v
        self._count[:n_full] += 1.0

    ### output ###

    def reduce(self, group=None):
        # Pool across the data-parallel group (same as the NLL accumulators). group=None is WORLD.
        for t in (self._sum, self._sqsum, self._count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)

    def save(self, path, seq_len):
        # Drop boundaries no sequence reached. boundary[i] = token position (state_chunk * (i+1)).
        keep = self._count.cpu().numpy() > 0
        boundary = ((np.arange(self._max_b) + 1) * self.state_chunk)[keep]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            boundary=boundary,
            layer=np.array(self.layer_ids),
            norm_sum=self._sum.cpu().numpy()[:, keep, :],
            norm_sqsum=self._sqsum.cpu().numpy()[:, keep, :],
            count=self._count.cpu().numpy()[keep],
            state_chunk=self.state_chunk,
            seq_len=seq_len,
        )


def install_state_norm_hooks(model, state_chunk, device):
    """Swap gated_delta_rule on every GatedDeltaNet with the segment-and-carry wrapper.

    Returns a StateNormAccumulator, or None if the model has no GatedDeltaNet layers (e.g. an
    attention baseline) -- in that case there is nothing to log and the caller skips it.
    """
    modules = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    if not modules:
        return None
    layer_ids = [m.layer_number for m in modules]
    num_heads = modules[0].num_v_heads_local_tp
    accum = StateNormAccumulator(state_chunk, layer_ids, num_heads, device)
    for m in modules:
        m.gated_delta_rule = _make_wrapper(m.gated_delta_rule, state_chunk, accum, m.layer_number)
    return accum


def _make_wrapper(real_fn, state_chunk, accum, layer_number):
    # gated_delta_rule is stored as a plain function on the instance and called without
    # self, so the wrapper takes the same positional q/k/v + keyword arguments and no self.
    def wrapper(query, key, value, *, g, beta, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        # Only the plain eval case is segmented. Packed sequences (cu_seqlens) and state
        # carry (initial_state) are passed straight through untouched, so this wrapper can
        # never perturb training or packed inference.
        if cu_seqlens is not None or initial_state is not None:
            return real_fn(query, key, value, g=g, beta=beta, initial_state=initial_state,
                           output_final_state=output_final_state,
                           use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)

        seq_len = query.shape[1]
        outs = []
        norms = []
        state = None
        start = 0
        while start < seq_len:
            end = min(start + state_chunk, seq_len)
            out_s, state = real_fn(
                query[:, start:end], key[:, start:end], value[:, start:end],
                g=g[:, start:end], beta=beta[:, start:end],
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=None,
            )
            outs.append(out_s)
            # Only record whole slices; skip the ragged tail (see module docstring).
            if end - start == state_chunk:
                # state [batch, heads, key_head_dim, value_head_dim] -> Frobenius over the
                # key_head_dim x value_head_dim matrix per head -> [batch, heads].
                norms.append(torch.linalg.vector_norm(state.float(), dim=(-2, -1)))
            start = end

        if norms:
            accum.record(layer_number, torch.stack(norms, dim=0))  # [num_full, batch, heads]
        core_attn_out = torch.cat(outs, dim=1)
        return core_attn_out, (state if output_final_state else None)

    return wrapper