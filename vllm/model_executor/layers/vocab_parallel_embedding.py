# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.logger import init_logger as _init_logger

_logger = _init_logger(__name__)

DEFAULT_VOCAB_PADDING_SIZE = 64


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
        self._maybe_prepare_xpu_int8_lm_head(layer)
        self._maybe_prepare_xpu_int4_draft_lm_head(layer)

    @staticmethod
    def _env_enabled(name: str) -> bool:
        return os.environ.get(name, "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _maybe_prepare_xpu_int8_lm_head(self, layer: torch.nn.Module) -> None:
        if not self._env_enabled("VLLM_XPU_LM_HEAD_INT8"):
            return
        if layer.__class__.__name__ != "ParallelLMHead":
            return
        scope = os.environ.get("VLLM_XPU_LM_HEAD_INT8_SCOPE", "all").strip().lower()
        prefix = str(getattr(layer, "_vllm_prefix", ""))
        prefix_parts = {part for part in prefix.replace("/", ".").split(".") if part}
        # Qwen3Next target head is loaded as "language_model.lm_head"; the
        # bundled MTP drafter remaps shared weights into a bare "lm_head".
        # Other MTP models include an explicit "mtp" prefix segment.
        is_mtp_head = ("mtp" in prefix_parts or prefix == "lm_head"
                            or getattr(layer, "_hx_is_draft_head", False))
        if is_mtp_head and self._env_enabled("VLLM_XPU_DRAFT_LM_HEAD_INT4"):
            return
        if scope in ("target", "target-only", "target_only") and is_mtp_head:
            return
        if scope in ("draft", "draft-only", "draft_only", "mtp") and not is_mtp_head:
            return
        if scope not in (
            "all",
            "target",
            "target-only",
            "target_only",
            "draft",
            "draft-only",
            "draft_only",
            "mtp",
        ):
            _logger.warning(
                "Unknown VLLM_XPU_LM_HEAD_INT8_SCOPE=%r; using all lm_head layers.",
                scope,
            )
        weight = getattr(layer, "weight", None)
        if weight is None or weight.device.type != "xpu":
            return
        if weight.dtype not in (torch.float16, torch.bfloat16):
            _logger.warning(
                "VLLM_XPU_LM_HEAD_INT8 requested but lm_head weight dtype is %s; "
                "falling back to the default BF16/FP16 path.",
                weight.dtype,
            )
            return
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401
        except Exception as exc:
            _logger.warning(
                "VLLM_XPU_LM_HEAD_INT8 requested but vllm_xpu_kernels._xpu_C "
                "could not be imported (%r); falling back to default lm_head.",
                exc,
            )
            return
        required_ops = ("int8_gemm_w8a8", "per_token_quant_int8_xpu")
        if any(not hasattr(torch.ops._xpu_C, op) for op in required_ops):
            _logger.warning(
                "VLLM_XPU_LM_HEAD_INT8 requested but required _xpu_C ops are "
                "unavailable; falling back to default lm_head.")
            return

        # Dense Qwen3.6 lm_head is BF16 even in the AutoRound INT4 checkpoint.
        # This experimental path keeps the original weight intact and adds a
        # transient per-output-channel INT8 copy. It is default-off and must pass
        # the quality gate before being treated as a candidate result.
        with torch.no_grad():
            # chunk the row loop — a single float() shot is a
            # 5 GB transient per head, which OOMs a single-card (TP1) load
            # where the head is unsharded (2.54 GB full vocab per card).
            num_rows, hidden = weight.shape
            chunk = int(
                os.environ.get("VLLM_XPU_LM_HEAD_INT8_CHUNK_ROWS", "4096")
                or "4096")
            scales = torch.empty(num_rows, dtype=torch.float32,
                                 device=weight.device)
            weight_q_t = torch.empty((hidden, num_rows), dtype=torch.int8,
                                     device=weight.device)
            for r0 in range(0, num_rows, chunk):
                r1 = min(r0 + chunk, num_rows)
                weight_f = weight[r0:r1].detach().float()
                row_scales = (
                    weight_f.abs().amax(dim=1).clamp_min(1.0e-10) / 127.0)
                scales[r0:r1] = row_scales
                weight_q_t[:, r0:r1] = (
                    torch.round(weight_f / row_scales.view(-1, 1))
                    .clamp(-127, 127)
                    .to(torch.int8)
                    .t()
                )
                del weight_f
            weight_q_t = weight_q_t.contiguous()
            scale_dtype = os.environ.get(
                "VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE", "fp32"
            ).strip().lower()
            if scale_dtype in ("bf16", "bfloat16"):
                scales = scales.to(torch.bfloat16)
            elif scale_dtype in ("fp16", "float16", "half"):
                scales = scales.to(torch.float16)
            elif scale_dtype not in ("", "fp32", "float32"):
                _logger.warning(
                    "Unknown VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE=%r; using fp32.",
                    scale_dtype,
                )
            scales = scales.contiguous()  # chunk-quantized above

        for name, tensor in (("_xpu_lm_head_int8_weight_t", weight_q_t),
                             ("_xpu_lm_head_int8_scale", scales)):
            if name in layer._buffers:
                layer._buffers[name] = tensor
            else:
                layer.register_buffer(name, tensor, persistent=False)
        _logger.info(
            "Prepared experimental XPU INT8 lm_head: prefix=%s scope=%s "
            "weight_t=%s scale=%s scale_dtype=%s",
            prefix or "<unknown>",
            scope or "all",
            tuple(weight_q_t.shape),
            tuple(scales.shape),
            scales.dtype,
        )

    def _maybe_prepare_xpu_int4_draft_lm_head(self, layer: torch.nn.Module) -> None:
        if not self._env_enabled("VLLM_XPU_DRAFT_LM_HEAD_INT4"):
            return
        if layer.__class__.__name__ != "ParallelLMHead":
            return
        prefix = str(getattr(layer, "_vllm_prefix", ""))
        prefix_parts = {part for part in prefix.replace("/", ".").split(".") if part}
        # Qwen3Next MTP draft heads are either a bare "lm_head" after sharing or
        # contain an "mtp" prefix segment. Keep the target lm_head exact.
        is_mtp_head = (getattr(layer, "_hx_is_draft_head", False)
                            or "mtp" in prefix_parts or prefix == "lm_head")
        if not is_mtp_head:
            return
        weight = getattr(layer, "weight", None)
        if weight is None or weight.device.type != "xpu":
            return
        if weight.dtype not in (torch.float16, torch.bfloat16):
            _logger.warning(
                "VLLM_XPU_DRAFT_LM_HEAD_INT4 requested but lm_head weight dtype "
                "is %s; falling back to the default draft lm_head.",
                weight.dtype,
            )
            return
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401
        except Exception as exc:
            _logger.warning(
                "VLLM_XPU_DRAFT_LM_HEAD_INT4 requested but "
                "vllm_xpu_kernels._xpu_C could not be imported (%r); falling "
                "back to default draft lm_head.",
                exc,
            )
            return
        if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
            _logger.warning(
                "VLLM_XPU_DRAFT_LM_HEAD_INT4 requested but "
                "_xpu_C.int4_gemm_w4a16 is unavailable; falling back.")
            return

        group_size = int(
            os.environ.get("VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE", "128")
            or "128")
        chunk_rows = int(
            os.environ.get("VLLM_XPU_DRAFT_LM_HEAD_INT4_CHUNK_ROWS", "2048")
            or "2048")
        if group_size <= 0 or weight.shape[1] % group_size != 0:
            raise ValueError(
                "VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE must divide hidden "
                f"size {weight.shape[1]}, got {group_size}")
        if group_size % 8 != 0:
            raise ValueError(
                "VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE must be divisible by 8")

        num_tokens, hidden = weight.shape
        packed_k = hidden // 8
        num_groups = hidden // group_size
        chunk_rows = max(1, min(chunk_rows, num_tokens))
        scale_dtype = os.environ.get(
            "VLLM_XPU_DRAFT_LM_HEAD_INT4_SCALE_DTYPE", "bf16"
        ).strip().lower()
        if scale_dtype in ("fp16", "float16", "half"):
            scale_torch_dtype = torch.float16
        elif scale_dtype in ("fp32", "float32"):
            scale_torch_dtype = torch.float32
        else:
            scale_torch_dtype = torch.bfloat16

        with torch.no_grad():
            # Store as [N, K/8] contiguous and pass the transposed view, matching
            # oneDNN's int4 NT layout requirement (logical [K/8, N] with
            # stride(0)==1). Values are unsigned nibbles with symmetric zp=8.
            packed_storage = torch.empty(
                (num_tokens, packed_k),
                dtype=torch.int32,
                device=weight.device,
            )
            scales = torch.empty(
                (num_groups, num_tokens),
                dtype=scale_torch_dtype,
                device=weight.device,
            )
            factors = (
                (16 ** torch.arange(8, device=weight.device, dtype=torch.int32))
                .view(1, 1, 8)
            )
            for start in range(0, num_tokens, chunk_rows):
                end = min(start + chunk_rows, num_tokens)
                w = weight[start:end].detach().float()
                w_grouped = w.view(end - start, num_groups, group_size)
                chunk_scales = (
                    w_grouped.abs().amax(dim=2).clamp_min(1.0e-10) / 7.0
                )
                scales[:, start:end] = chunk_scales.t().to(scale_torch_dtype)
                q = torch.round(
                    w_grouped / chunk_scales.unsqueeze(-1)
                ).clamp(-8, 7).to(torch.int32) + 8
                packed = (
                    (q.view(end - start, packed_k, 8) * factors)
                    .sum(dim=2)
                    .to(torch.int32)
                )
                packed_storage[start:end].copy_(packed)
                del w, w_grouped, chunk_scales, q, packed

            qweight = packed_storage.t()
            qzeros = torch.tensor([8], dtype=torch.int8, device=weight.device)

        for name, tensor in (
            ("_xpu_lm_head_int4_weight_t", qweight),
            ("_xpu_lm_head_int4_scale", scales.contiguous()),
            ("_xpu_lm_head_int4_qzeros", qzeros),
        ):
            if name in layer._buffers:
                layer._buffers[name] = tensor
            else:
                layer.register_buffer(name, tensor, persistent=False)
        layer._xpu_lm_head_int4_group_size = group_size
        _logger.info(
            "Prepared experimental XPU INT4 draft lm_head: prefix=%s "
            "weight_t=%s stride=%s scale=%s scale_dtype=%s group_size=%d",
            prefix or "<unknown>",
            tuple(qweight.shape),
            tuple(qweight.stride()),
            tuple(scales.shape),
            scales.dtype,
            group_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            return linear_batch_invariant(x, layer.weight, bias)
        if (
            os.environ.get("VLLM_XPU_DRAFT_LM_HEAD_INT4", "0").strip().lower()
            in ("1", "true", "yes", "on")
            and bias is None
            and x.device.type == "xpu"
            and hasattr(layer, "_xpu_lm_head_int4_weight_t")
            and hasattr(layer, "_xpu_lm_head_int4_scale")
            and hasattr(layer, "_xpu_lm_head_int4_qzeros")
        ):
            # draft lm_head as RTN INT4 g128
            # through oneDNN int4_gemm_w4a16. Draft logits only steer token
            # proposals; the FP16/INT8 target head still verifies.
            x_contiguous = x if x.is_contiguous() else x.contiguous()
            out_shape = x_contiguous.shape[:-1] + (
                layer._xpu_lm_head_int4_weight_t.shape[1],
            )
            logits = torch.ops._xpu_C.int4_gemm_w4a16(
                x_contiguous.reshape(-1, x_contiguous.shape[-1]),
                layer._xpu_lm_head_int4_weight_t,
                None,
                layer._xpu_lm_head_int4_scale,
                layer._xpu_lm_head_int4_qzeros,
                layer._xpu_lm_head_int4_group_size,
                None,
            )
            fallback_margin_s = os.environ.get(
                "VLLM_XPU_DRAFT_LM_HEAD_INT4_FALLBACK_MARGIN", "0")
            try:
                fallback_margin = float(fallback_margin_s or "0")
            except ValueError:
                fallback_margin = 0.0
            if fallback_margin > 0.0 and logits.shape[-1] >= 2:
                top2 = torch.topk(logits.float(), k=2, dim=-1).values
                low_margin = (top2[:, 0] - top2[:, 1]) < fallback_margin
                if bool(low_margin.any().item()):
                    flat_x = x_contiguous.reshape(
                        -1, x_contiguous.shape[-1])
                    exact_logits = F.linear(
                        flat_x[low_margin].to(layer.weight.dtype),
                        layer.weight,
                        bias,
                    )
                    logits[low_margin] = exact_logits.to(logits.dtype)
            return logits.reshape(out_shape)
        if (
            os.environ.get("VLLM_XPU_LM_HEAD_INT8", "0").strip().lower()
            in ("1", "true", "yes", "on")
            and bias is None
            and x.device.type == "xpu"
            and hasattr(layer, "_xpu_lm_head_int8_weight_t")
            and hasattr(layer, "_xpu_lm_head_int8_scale")
        ):
            # W8A8 INT8 lm_head: the 2.54 GB
            # FP16 vocab GEMM is read on every draft step and every verify;
            # INT8 halves the single biggest per-step weight read.
            x_contiguous = x if x.is_contiguous() else x.contiguous()
            x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(
                x_contiguous)
            return torch.ops._xpu_C.int8_gemm_w8a8(
                x_q,
                x_scale,
                layer._xpu_lm_head_int8_weight_t,
                layer._xpu_lm_head_int8_scale,
                layer.weight.dtype,
                None,
            )
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    def tie_weights(
        self, layer: torch.nn.Module, embed_tokens: "VocabParallelEmbedding"
    ):
        layer.weight = embed_tokens.weight
        return layer


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# --8<-- [start:vocab_parallel_embedding]
@PluggableLayer.register("vocab_parallel_embedding")
class VocabParallelEmbedding(PluggableLayer):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1025 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1010 | ... | 1025 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  528 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  | 528 | ... |  543 |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        quant_config: quant config for the layer
        prefix: full name of the layer in the state dict
        disable_tp: If true, tensor parallelism will be disabled for this layer.
    """  # noqa: E501

    # --8<-- [end:vocab_parallel_embedding]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        super().__init__()

        # Keep the input dimensions.
        self.disable_tp = disable_tp
        if disable_tp:
            tp_rank, self.tp_size = 0, 1
        else:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = not isinstance(self, ParallelLMHead)
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # Divide the weight matrix along the vocabulary dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )
        self.update_param_tp_status()

    def update_param_tp_status(self):
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def get_sharded_to_full_mapping(self) -> list[int] | None:
        """Get a mapping that can be used to reindex the gathered
        logits for sampling.

        During sampling, we gather logits from all ranks. The relationship
        of index->token_id will follow the same format as outlined in the class
        docstring. However, after the gather, we want to reindex the final
        logits tensor to map index->token_id one-to-one (the index is always
        equal the token_id it corresponds to). The indices returned by this
        method allow us to do that.
        """
        if self.tp_size < 2:
            return None

        base_embeddings: list[int] = []
        added_embeddings: list[int] = []
        padding: list[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            if (
                loaded_weight.ndim == 0
                and param.data.ndim == 1
                and param.data.numel() == 1
            ):
                loaded_weight = loaded_weight.reshape(1)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = (
                param.packed_factor
                if isinstance(param, BasevLLMParameter)
                else param.pack_factor
            )
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == self.org_vocab_size

        # Copy the data. Select chunk corresponding to current shard.
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            # Reduce across all the model parallel GPUs.
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings}"
        s += f", num_embeddings_per_partition={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


# --8<-- [start:parallel_lm_head]
@PluggableLayer.register("parallel_lm_head")
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        disable_tp: If true, tensor parallelism will be disabled for this layer.
    """

    # --8<-- [end:parallel_lm_head]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
            disable_tp=disable_tp,
        )
        self.quant_config = quant_config
        if bias:
            self._register_bias()
        else:
            self.register_parameter("bias", None)

    def _register_bias(self):
        data = torch.empty(self.num_embeddings_per_partition, dtype=self.params_dtype)
        self.bias = Parameter(data, requires_grad=False)
        weight_attrs = dict(output_dim=0, weight_loader=self.weight_loader)
        set_weight_attrs(weight=self.bias, weight_attrs=weight_attrs)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        return self.quant_method.tie_weights(self, embed_tokens)

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
