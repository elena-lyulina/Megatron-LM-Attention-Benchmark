# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025-2026, Songlin Yang, Yu Zhang, Zhiyuan Li (FLA authors).

# The Kimi Delta Attention algorithm implemented here follows
#   "Kimi Linear: An Expressive, Efficient Attention Architecture"
#   https://arxiv.org/abs/2510.26692
# and uses the `chunk_kda` operator from flash-linear-attention (FLA) as the
# numerical reference, mirroring the structure of ``gated_delta_net.py``.

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.modules.convolution import causal_conv1d, causal_conv1d_update
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    causal_conv1d_update = None
    chunk_kda = None
    fused_recurrent_kda = None

    HAVE_FLA = False

logger = logging.getLogger(__name__)


@dataclass
class KimiDeltaAttentionSubmodules:
    """Module specs for the projections and output norm of a KimiDeltaAttention layer.

    KDA keeps separate ``q``/``k``/``v``/``beta`` projections (like the FLA reference) rather
    than a single fused input projection, and adds two low-rank bottlenecks: ``f_proj`` for the
    per-channel forget gate and ``g_proj`` for the output gate.
    """

    q_proj: Union[ModuleSpec, type] = IdentityOp
    k_proj: Union[ModuleSpec, type] = IdentityOp
    v_proj: Union[ModuleSpec, type] = IdentityOp
    b_proj: Union[ModuleSpec, type] = IdentityOp
    f_proj_down: Union[ModuleSpec, type] = IdentityOp
    f_proj_up: Union[ModuleSpec, type] = IdentityOp
    g_proj_down: Union[ModuleSpec, type] = IdentityOp
    g_proj_up: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class KimiDeltaAttention(MegatronModule):
    """Kimi Delta Attention (KDA) layer.

    KDA refines the gated delta rule with a *fine-grained* (per-key-channel) forget gate. Where
    Gated DeltaNet uses a scalar per-head decay, KDA uses ``Diag(alpha_t)`` with ``alpha_t`` in
    ``[0, 1]^{d_k}`` (Eq. 1 of https://arxiv.org/abs/2510.26692)::

        S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T ,   o_t = S_t^T q_t

    The recurrence, chunk-wise parallel form, q/k L2-normalization, beta sigmoid and the
    ``-exp(A_log) * softplus(f_proj(x) + dt_bias)`` decay are all evaluated inside the FLA
    ``chunk_kda`` kernel. This module owns the projections, the short convolution, the decay
    parameters (``A_log``, ``dt_bias``) and the gated output RMSNorm.

    Takes input of shape ``[s, b, h]`` and returns output of the same shape.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        A_init_range: tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        name: str | None = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Module specs for the projections and output norm.
            layer_number: The layer number of this KDA layer.
            bias: Whether to use bias in the linear projections.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: Initialization range for the causal convolution weights.
            A_init_range: Initialization range for the per-head decay strength ``A``.
            pg_collection: Process groups for tensor model parallelism.
            name: Module instance name passed top-down from the parent module.
        """

        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        assert pg_collection is not None, "pg_collection must be provided for KimiDeltaAttention"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.cp_size = self.pg_collection.cp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        # Low-rank bottleneck dimension of the f_proj / g_proj gates (FLA uses head_v_dim).
        self.gate_low_rank_dim = self.value_head_dim

        assert self.cp_size == 1, "KimiDeltaAttention does not support context parallelism yet."
        assert self.activation in [
            "silu",
            "swish",
        ], f"KimiDeltaAttention only supports silu/swish activation, got {self.activation}."
        assert not self.config.deterministic_mode, (
            "KimiDeltaAttention does not support deterministic mode yet "
            "(the chunk_kda kernel is non-deterministic)."
        )

        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        # Per-value-head, per-key-channel gate (Diag(alpha_t)). See Eq. 1 in the paper.
        self.gate_dim = self.key_head_dim * self.num_value_heads

        self.num_key_heads_local_tp = self.num_key_heads // self.tp_size
        self.num_value_heads_local_tp = self.num_value_heads // self.tp_size
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size
        self.gate_dim_local_tp = self.gate_dim // self.tp_size

        # q/k/v/beta projections (column parallel, sharded over heads).
        self.q_proj = self._build_column_parallel(
            submodules.q_proj, self.hidden_size, self.qk_dim, "q_proj", name
        )
        self.k_proj = self._build_column_parallel(
            submodules.k_proj, self.hidden_size, self.qk_dim, "k_proj", name
        )
        self.v_proj = self._build_column_parallel(
            submodules.v_proj, self.hidden_size, self.v_dim, "v_proj", name
        )
        self.b_proj = self._build_column_parallel(
            submodules.b_proj, self.hidden_size, self.num_value_heads, "b_proj", name
        )

        # Fine-grained forget gate f_proj: low-rank bottleneck (Diag(alpha_t) pre-activation).
        # The down-projection is replicated across TP (duplicated), the up-projection is sharded.
        self.f_proj_down = build_module(
            submodules.f_proj_down,
            self.hidden_size,
            self.gate_low_rank_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            is_expert=False,
            parallel_mode="duplicated",
            tp_comm_buffer_name="f_proj_down",
            tp_group=None,
            name=(name + ".f_proj_down") if name is not None else None,
        )
        self.f_proj_up = self._build_column_parallel(
            submodules.f_proj_up, self.gate_low_rank_dim, self.gate_dim, "f_proj_up", name
        )

        # Output gate g_proj: low-rank bottleneck, gates the output RMSNorm.
        self.g_proj_down = build_module(
            submodules.g_proj_down,
            self.hidden_size,
            self.gate_low_rank_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            is_expert=False,
            parallel_mode="duplicated",
            tp_comm_buffer_name="g_proj_down",
            tp_group=None,
            name=(name + ".g_proj_down") if name is not None else None,
        )
        self.g_proj_up = self._build_column_parallel(
            submodules.g_proj_up, self.gate_low_rank_dim, self.v_dim, "g_proj_up", name
        )

        # Depthwise short convolution over concatenated q/k/v.
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        setattr(self.conv1d.weight, "partition_dim", 0)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)

        # Decay parameters. A_log is per value-head; dt_bias is per gate channel (HV * K).
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_value_heads_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.gate_dim_local_tp, dtype=torch.float32, device=torch.cuda.current_device()
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)

        # Gated output RMSNorm (applied per value-head channel), then output projection.
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )
        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
            name=(name + ".out_proj") if name is not None else None,
        )

        self.reset_parameters()

    def _build_column_parallel(self, spec, input_dim, output_dim, buffer_name, name):
        """Build a column-parallel linear projection sharded over its output (head) dimension."""
        return build_module(
            spec,
            input_dim,
            output_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=buffer_name,
            tp_group=self.pg_collection.tp,
            name=(name + "." + buffer_name) if name is not None else None,
        )

    def reset_parameters(self):
        """Initialize the convolution and decay parameters (Section 4 of the paper)."""
        if not self.config.perform_initialization:
            return
        with get_cuda_rng_tracker().fork():
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
            # A_log = log(A), A ~ U(A_init_range), per value-head.
            A = torch.empty(
                self.num_value_heads_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            ).uniform_(*self.A_init_range)
            self.A_log.data.copy_(torch.log(A))
            # dt_bias = inverse-softplus of dt ~ exp(U(log 1e-3, log 0.1)), per gate channel.
            dt = torch.exp(
                torch.rand(
                    self.gate_dim_local_tp, dtype=torch.float32, device=torch.cuda.current_device()
                )
                * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
                + torch.log(torch.tensor(0.001))
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.data.copy_(inv_dt)

    def _resolve_cu_seqlens(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name):
        """Pick padded over actual cu_seqlens and check it spans the whole packed sequence.

        Mirrors ``GatedDeltaNet._resolve_cu_seqlens`` (minus the cp_size check, KDA is cp=1).
        """
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        total_cu = cu_seqlens[-1].cpu().item()
        if total_cu != total_seq_len:
            raise ValueError(
                f"KDA: {name}[-1]={total_cu} does not match total_sequence_length={total_seq_len} "
                f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
            )
        return cu_seqlens

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """Forward pass of the KDA layer.

        Args:
            hidden_states (Tensor): Input of shape ``[s, b, h]`` (already input-layernormed).
            attention_mask (Tensor): Unused (KDA is causal by construction).
            inference_context (Optional[BaseInferenceContext]): KV-cache context (unsupported).
            packed_seq_params (Optional[PackedSeqParams]): THD packing params. Supported: the
                per-document ``cu_seqlens`` are passed to the conv and the recurrence so state
                is reset at every document boundary.
            sequence_len_offset (Optional[int]): Inference CUDA-graph offset (unsupported).

        Returns:
            tuple[Tensor, Optional[Tensor]]: KDA output ``[s, b, h]`` and the output-proj bias.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        _, batch, _ = hidden_states.shape

        prefill_inference = False
        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "KDA does not currently support dynamic inference batching."
            assert (
                not self.config.sequence_parallel
            ), "KDA inference does not currently support sequence parallelism."
            assert self.cp_size == 1, "KDA inference does not currently support context parallel."
            assert not self.config.deterministic_mode, (
                "KDA inference requires the fast (non-deterministic) FLA kernels."
            )
            assert self.config.pipeline_model_parallel_size == 1, (
                "KDA inference is only validated for pipeline_model_parallel_size == 1."
            )
            if inference_context.sequence_len_offset > 0:
                # Decode: one token at a time, served from the cached conv + recurrent state.
                return self._decode(hidden_states, inference_context)
            # Prefill: fall through the normal forward, capturing the final conv + recurrent state
            # into the cache (output_final_state flags + cache write below).
            prefill_inference = True

        # Projections (s b x). q/k/v/beta live on their own column-parallel GEMMs.
        nvtx_range_push(suffix="in_proj")
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        beta, _ = self.b_proj(hidden_states)
        g_raw, _ = self.f_proj_up(self.f_proj_down(hidden_states)[0])
        z, _ = self.g_proj_up(self.g_proj_down(hidden_states)[0])
        nvtx_range_pop(suffix="in_proj")

        # s b x -> b s x
        q, k, v, beta, g_raw, z = (t.transpose(0, 1) for t in (q, k, v, beta, g_raw, z))
        seq_len = q.shape[1]

        # Packed (THD) sequences: the microbatch is flattened to [T, 1, x] and cu_seqlens marks
        # the document boundaries. Passing it to the conv + recurrence resets state per document.
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "KDA packed (THD) expects batch dim 1"
            assert not self.config.deterministic_mode, \
                "KDA packed sequence does not support deterministic mode"
            cu_seqlens = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len,
                "cu_seqlens_q",
            )
        else:
            cu_seqlens = None

        # Depthwise short conv (with silu) over concatenated q/k/v.
        nvtx_range_push(suffix="conv1d")
        qkv = torch.cat([q, k, v], dim=-1)
        qkv, conv_state = causal_conv1d(
            x=qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias if self.conv_bias else None,
            activation=self.activation,
            initial_state=None,
            output_final_state=prefill_inference,
            cu_seqlens=cu_seqlens,
        )
        q, k, v = torch.split(
            qkv, [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp], dim=-1
        )
        nvtx_range_pop(suffix="conv1d")

        # Reshape to head layout. q/k at key heads; v/g/beta at value heads (GVA handled by kernel).
        q = q.reshape(batch, seq_len, self.num_key_heads_local_tp, self.key_head_dim).contiguous()
        k = k.reshape(batch, seq_len, self.num_key_heads_local_tp, self.key_head_dim).contiguous()
        v = v.reshape(
            batch, seq_len, self.num_value_heads_local_tp, self.value_head_dim
        ).contiguous()
        g_raw = g_raw.reshape(
            batch, seq_len, self.num_value_heads_local_tp, self.key_head_dim
        ).contiguous()
        beta = beta.reshape(batch, seq_len, self.num_value_heads_local_tp).contiguous()

        # KDA recurrence. L2-norm(q,k), beta sigmoid and the fine-grained decay
        # g = -exp(A_log) * softplus(g_raw + dt_bias) are all fused inside the kernel.
        nvtx_range_push(suffix="chunk_kda")
        core_attn_out, recurrent_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g_raw,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=prefill_inference,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        nvtx_range_pop(suffix="chunk_kda")

        if prefill_inference:
            # Stash the prefill's final states; _decode advances them one token at a time. Storing
            # the kernel outputs directly avoids any pre-allocated shape assumptions. Greedy only:
            # beam search's swap_key_value_dict assumes attention K/V (batch on dim 1), but these
            # states have batch on dim 0.
            inference_context.key_value_memory_dict[self.layer_number] = (
                conv_state.detach(),
                recurrent_state.detach(),
            )

        # Gated output RMSNorm: RMSNorm(o) * sigmoid(z), per value-head channel.
        nvtx_range_push(suffix="gated_norm")
        z = z.reshape(batch, seq_len, self.num_value_heads_local_tp, self.value_head_dim)
        norm_out = self._apply_gated_norm(core_attn_out, z)
        nvtx_range_pop(suffix="gated_norm")

        # b s x -> s b x, then output projection.
        norm_out = norm_out.reshape(batch, seq_len, -1).transpose(0, 1).contiguous()
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    def _decode(self, hidden_states, inference_context):
        """Single-token incremental decode.

        Advances the cached conv + recurrent state by one token using the FLA single-step kernels.
        Mirrors the prefill forward but with one token and the cached state as input. Static
        batching, CP=1, non-deterministic kernels (all asserted in forward before this is reached).

        Args:
            hidden_states (Tensor): ``[1, b, h]`` — the single new token's hidden state.
            inference_context (BaseInferenceContext): holds the per-layer (conv_state, recurrent_state).

        Returns:
            tuple[Tensor, Optional[Tensor]]: KDA output ``[1, b, h]`` and the output-proj bias.
        """
        assert self.layer_number in inference_context.key_value_memory_dict, (
            "KDA _decode called before prefill populated the state cache."
        )
        seq_len, batch, _ = hidden_states.shape
        assert seq_len == 1, "KDA decode processes one token at a time"
        conv_state, recurrent_state = inference_context.key_value_memory_dict[self.layer_number]
        assert recurrent_state.shape[0] == batch, "KDA decode batch must match prefill batch."

        # Projections (s b x), then s b x -> b s x (s == 1). CP=1, so no all-to-all.
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        beta, _ = self.b_proj(hidden_states)
        g_raw, _ = self.f_proj_up(self.f_proj_down(hidden_states)[0])
        z, _ = self.g_proj_up(self.g_proj_down(hidden_states)[0])
        q, k, v, beta, g_raw, z = (t.transpose(0, 1) for t in (q, k, v, beta, g_raw, z))

        # Causal conv single step. Updates conv_state in place; same [N, D, W] cache layout the
        # prefill causal_conv1d(output_final_state=True) produced.
        qkv = torch.cat([q, k, v], dim=-1)
        qkv, conv_state = causal_conv1d_update(
            x=qkv,
            cache=conv_state,
            weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
            bias=self.conv1d.bias if self.conv_bias else None,
            activation=self.activation,
        )
        q, k, v = torch.split(
            qkv, [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp], dim=-1
        )

        # Head layout (mirror forward): q/k at key heads, v/g at value heads, beta scalar per head.
        q = q.reshape(batch, 1, self.num_key_heads_local_tp, self.key_head_dim).contiguous()
        k = k.reshape(batch, 1, self.num_key_heads_local_tp, self.key_head_dim).contiguous()
        v = v.reshape(batch, 1, self.num_value_heads_local_tp, self.value_head_dim).contiguous()
        g_raw = g_raw.reshape(
            batch, 1, self.num_value_heads_local_tp, self.key_head_dim
        ).contiguous()
        beta = beta.reshape(batch, 1, self.num_value_heads_local_tp).contiguous()

        # Recurrent single step. Same raw g/beta + in-kernel decay/sigmoid/L2-norm convention as
        # chunk_kda, so it is numerically consistent with the prefill; T == 1.
        core_attn_out, recurrent_state = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g_raw,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        inference_context.key_value_memory_dict[self.layer_number] = (conv_state, recurrent_state)

        # Gated output RMSNorm + output projection. b s x -> s b x (CP=1, so no all-to-all).
        z = z.reshape(batch, 1, self.num_value_heads_local_tp, self.value_head_dim)
        norm_out = self._apply_gated_norm(core_attn_out, z)
        norm_out = norm_out.reshape(batch, 1, -1).transpose(0, 1).contiguous()
        out, out_bias = self.out_proj(norm_out)
        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        """Gated output RMSNorm: ``RMSNorm(x) * sigmoid(gate)`` (FLA FusedRMSNormGated)."""
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * torch.sigmoid(gate.float())
        return y.to(x_dtype)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Sharded state dict for distributed checkpointing.

        The child projections shard themselves; here we add TP sharding for the raw parameters
        (``A_log``, ``dt_bias``) and the depthwise ``conv1d``, splitting the conv weight into its
        q/k/v sections so the layout is stable across TP world sizes.
        """
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp

        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )

        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict

    def backward_dw(self):
        """Execute weight gradient computation for the linear projections."""
        for module in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.b_proj,
            self.f_proj_down,
            self.f_proj_up,
            self.g_proj_down,
            self.g_proj_up,
            self.out_proj,
        ):
            if hasattr(module, "backward_dw"):
                module.backward_dw()
