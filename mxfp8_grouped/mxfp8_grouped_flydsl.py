"""Route ``aten::_scaled_grouped_mm_v2`` MXFP8 to a FlyDSL kernel on ROCm.

On ROCm the MXFP8 arm of ``_scaled_grouped_mm_v2`` is compiled out
(``aten/src/ATen/native/hip/GroupedBlas.cpp:171`` guards the MSLK call with
``!defined(USE_ROCM)``), so every ``BlockWise1x32`` call raises
NotImplementedError. MSLK's ``mx8mx8bf16_grouped`` kernels are CUTLASS-only;
its ROCm arm only builds ``fp8_rowwise_grouped``.

FlyDSL kernels JIT from Python, so that C++ arm cannot launch one. Instead we
replace the op's CUDA kernel from Python. That needs no PyTorch rebuild, and it
keeps existing MoE code working unchanged: callers still go through
``torch._scaled_grouped_mm_v2``.

Usage::

    import mxfp8_grouped_flydsl as mxg
    import flydsl_grouped

    mxg.register_kernel(flydsl_grouped.kernel)  # or your own; see MXFP8GroupedKernel
    mxg.register()

``flydsl_grouped.kernel`` is the basic FlyDSL implementation. Leaving the kernel
unset falls back to a slow dequantize-and-matmul reference.

Everything that is not MXFP8 is forwarded to ``aten::_scaled_grouped_mm`` (v1),
which is a different op and so is unaffected by the override. On ROCm v2's only
other reachable recipe is rowwise, so that forward is complete coverage; the
fp4 recipes are CUDA-only and are rejected with the message the C++ would give.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch


# ATen ScalingType (aten/src/ATen/BlasBackend.h:33) and SwizzleType (:42).
TENSORWISE = 0
ROWWISE = 1
BLOCKWISE_1X32 = 3

NO_SWIZZLE = 0
SWIZZLE_32_4_4 = 1

BLOCK_SIZE = 32

# Scale rows are padded to 128 and scale columns (K/32) to 4, per
# _check_scales_blocked in hip/GroupedBlas.cpp:415.
SCALE_ROW_ALIGN = 128
SCALE_COL_ALIGN = 4


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@dataclass(frozen=True)
class MXFP8GroupedProblem:
    """Shape metadata for one grouped MXFP8 GEMM call.

    ``case`` is "2d3d" (ragged M, static per-expert weights - the MoE case) or
    "2d2d" (ragged K). ``blocked_k``/``blocked_n`` are the padded scale extents
    the ATen contract requires; the logical extents are ``k // BLOCK_SIZE``
    and ``n``.
    """

    case: str
    g: int
    m: int
    n: int
    k: int
    blocked_k: int
    blocked_n: int


# A kernel takes fp8_e4m3fn operands plus e8m0 scales and writes bf16 into
# ``out`` in place. Layouts are exactly what ATen hands us, described in
# _validate_mxfp8 below; ``offs`` is the int32 cumulative-end vector on device.
MXFP8GroupedKernel = Callable[
    [
        torch.Tensor,  # mat_a
        torch.Tensor,  # mat_b
        torch.Tensor,  # scale_a
        torch.Tensor,  # scale_b
        torch.Tensor,  # offs
        torch.Tensor,  # out
        MXFP8GroupedProblem,
    ],
    None,
]

_kernel: Optional[MXFP8GroupedKernel] = None

# Set MXFP8_GROUPED_FORCE_REFERENCE=1 to bypass a registered kernel and use the
# slow dequant reference, for A/B correctness checks.
_FORCE_REFERENCE = os.environ.get("MXFP8_GROUPED_FORCE_REFERENCE", "0") == "1"


def register_kernel(fn: Optional[MXFP8GroupedKernel]) -> None:
    global _kernel
    _kernel = fn


def registered_kernel() -> Optional[MXFP8GroupedKernel]:
    return _kernel


def _is_mxfp8(mat_a, mat_b, scale_a, recipe_a, scale_b, recipe_b) -> bool:
    """Mirror scaled_blas::check_mxfp8_recipe (ScaledBlasUtils.cpp:212)."""
    return (
        mat_a.dtype == torch.float8_e4m3fn
        and mat_b.dtype == torch.float8_e4m3fn
        and len(scale_a) == 1
        and len(scale_b) == 1
        and len(recipe_a) == 1
        and len(recipe_b) == 1
        and recipe_a[0] == BLOCKWISE_1X32
        and recipe_b[0] == BLOCKWISE_1X32
        and scale_a[0].dtype == torch.float8_e8m0fnu
        and scale_b[0].dtype == torch.float8_e8m0fnu
    )


def _validate_mxfp8(mat_a, mat_b, scale_a, swizzle_a, scale_b, swizzle_b, offs, bias, out_dtype):
    """Reproduce the checks _scaled_grouped_mm_cuda_v2 would have run.

    Layouts (hip/GroupedBlas.cpp:396-443, :646-692):
      2d-3d: mat_a (total_M, K) row-major, mat_b (G, K, N) with stride(-2) == 1,
             scale_a 2D with rows >= total_M and cols >= K/32 (padded to
             128 / 4), scale_b 2D (G, blocked_k * blocked_n) holding each
             expert's (N, K/32) scale block, out (total_M, N).
      2d-2d: mat_a (M, total_K), mat_b (total_K, N), offs splits K,
             out (G, M, N).
    """
    if bias is not None:
        raise ValueError("Bias not supported yet")
    if out_dtype not in (None, torch.bfloat16):
        raise ValueError("Only bf16 high precision output types are supported for grouped gemm")
    if mat_a.dim() != 2:
        raise ValueError("MXFP8 grouped GEMM currently only supports 2d-2d and 2d-3d cases")
    if mat_b.dim() not in (2, 3):
        raise ValueError("MXFP8 grouped GEMM currently only supports 2d-2d and 2d-3d cases")
    if offs is None:
        raise ValueError("MXFP8 2d-2d and 2d-3d grouped GEMMs requires offsets")
    if offs.dim() != 1 or offs.dtype != torch.int32:
        raise ValueError("offs has to be 1D and int32")
    if swizzle_a[0] != NO_SWIZZLE or swizzle_b[0] != NO_SWIZZLE:
        raise ValueError("For ROCM MXFP8 grouped gemm, both scale swizzle types must be SWIZZLE_NONE")
    if mat_a.stride(-1) != 1:
        raise ValueError("Expected mat1 to not be transposed")
    if mat_b.stride(-2) != 1:
        raise ValueError("Expected mat2 to be transposed")

    k = mat_a.size(-1)
    if k % BLOCK_SIZE != 0:
        raise ValueError(f"K must be a multiple of {BLOCK_SIZE} for MXFP8, got {k}")

    if mat_b.dim() == 3:
        g, kb, n = mat_b.shape
        if kb != k:
            raise ValueError("contraction dimension of mat_a and mat_b must match")
        if offs.size(0) != g:
            raise ValueError("matrix batch sizes have to match")
        case, m = "2d3d", mat_a.size(0)
        blocked_k = _round_up(k // BLOCK_SIZE, SCALE_COL_ALIGN)
        blocked_n = _round_up(n, SCALE_ROW_ALIGN)
        if scale_b.dim() != 2 or scale_b.shape != (g, blocked_k * blocked_n):
            raise ValueError(
                f"for block-scaled grouped GEMM, the tensor shape ({g}, {k}, {n}) must have "
                f"scale shape ({g},{blocked_k},{blocked_n}) for arg 1, got: {tuple(scale_b.shape)}"
            )
    else:
        n = mat_b.size(1)
        case, g, m = "2d2d", offs.size(0), mat_a.size(0)
        blocked_k = _round_up(k // BLOCK_SIZE, SCALE_COL_ALIGN)
        blocked_n = _round_up(n, SCALE_ROW_ALIGN)
        if scale_b.dim() != 2 or scale_b.size(0) < n:
            raise ValueError(
                f"for block-scaled, arg 1 tensor shape {tuple(mat_b.shape)} must have "
                f"scale.shape[0] >= {n} but got scale.shape={tuple(scale_b.shape)}"
            )

    if scale_a.dim() != 2 or scale_a.size(0) < m:
        raise ValueError(
            f"for block-scaled, arg 0 tensor shape {tuple(mat_a.shape)} must have "
            f"scale.shape[0] >= {m} but got scale.shape={tuple(scale_a.shape)}"
        )

    return MXFP8GroupedProblem(case, g, m, n, k, blocked_k, blocked_n)


def _dequant(mat: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """mat (R, C) fp8 times scale (R, C/BLOCK_SIZE) e8m0, in fp32."""
    return mat.float() * scale.float().repeat_interleave(BLOCK_SIZE, dim=1)


def _reference(mat_a, mat_b, scale_a, scale_b, offs, out, p: MXFP8GroupedProblem) -> None:
    """Slow dequant-and-matmul reference, so the dispatch path is testable
    before a real kernel lands. Not intended to be fast."""
    kb = p.k // BLOCK_SIZE
    start = 0
    scol = 0  # ragged-K scales are padded per group, so track the column offset
    for g, end in enumerate(offs.tolist()):
        if p.case == "2d3d":
            a = _dequant(mat_a[start:end], scale_a[start:end, :kb])
            b = _dequant(mat_b[g].t(), scale_b[g].view(p.blocked_n, p.blocked_k)[: p.n, :kb])
            out[start:end] = (a @ b.t()).to(out.dtype)
        else:
            kbg = (end - start) // BLOCK_SIZE
            a = _dequant(mat_a[:, start:end], scale_a[: p.m, scol : scol + kbg])
            b = _dequant(mat_b[start:end].t(), scale_b[: p.n, scol : scol + kbg])
            out[g] = (a @ b.t()).to(out.dtype)
            scol += _round_up(kbg, SCALE_COL_ALIGN)
        start = end


def _mxfp8_grouped_mm(mat_a, mat_b, scale_a, swizzle_a, scale_b, swizzle_b, offs, bias, out_dtype):
    p = _validate_mxfp8(mat_a, mat_b, scale_a, swizzle_a, scale_b, swizzle_b, offs, bias, out_dtype)
    out_shape = (p.m, p.n) if p.case == "2d3d" else (p.g, p.m, p.n)
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=mat_a.device)

    if _kernel is not None and not _FORCE_REFERENCE:
        _kernel(mat_a, mat_b, scale_a, scale_b, offs, out, p)
    else:
        _reference(mat_a, mat_b, scale_a, scale_b, offs, out, p)
    return out


def _scaled_grouped_mm_v2_impl(
    mat_a,
    mat_b,
    scale_a: Sequence[torch.Tensor],
    recipe_a: Sequence[int],
    swizzle_a: Sequence[int],
    scale_b: Sequence[torch.Tensor],
    recipe_b: Sequence[int],
    swizzle_b: Sequence[int],
    offs=None,
    bias=None,
    out_dtype=None,
    contraction_dim=(),
    use_fast_accum=False,
):
    if _is_mxfp8(mat_a, mat_b, scale_a, recipe_a, scale_b, recipe_b):
        return _mxfp8_grouped_mm(
            mat_a, mat_b, scale_a[0], swizzle_a, scale_b[0], swizzle_b, offs, bias, out_dtype
        )

    if list(recipe_a) in ([ROWWISE], [TENSORWISE]) and list(recipe_b) in ([ROWWISE], [TENSORWISE]):
        # v1 is a distinct op, so this does not re-enter the override. Verified
        # bitwise identical to the un-overridden v2 rowwise path on gfx950.
        return torch.ops.aten._scaled_grouped_mm(
            mat_a,
            mat_b,
            scale_a[0],
            scale_b[0],
            offs,
            bias,
            None,
            out_dtype or torch.bfloat16,
            use_fast_accum,
        )

    raise NotImplementedError(
        f"_scaled_grouped_mm_v2: recipe_a={list(recipe_a)} recipe_b={list(recipe_b)} is not "
        "supported on ROCm (mxfp4/nvfp4 grouped GEMM is CUDA-only)"
    )


def _scaled_grouped_mm_v2_meta(
    mat_a, mat_b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
    offs=None, bias=None, out_dtype=None, contraction_dim=(), use_fast_accum=False,
):
    """v2 ships no Meta kernel, so torch.compile cannot trace it. Supplying one
    mirrors create_grouped_gemm_output_tensor (GroupedMMUtils.h:47)."""
    a_is_2d, b_is_2d = mat_a.dim() == 2, mat_b.dim() == 2
    if a_is_2d:
        size = (offs.size(0), mat_a.size(0), mat_b.size(1)) if b_is_2d else (mat_a.size(0), mat_b.size(-1))
    elif b_is_2d:
        size = (mat_a.size(1), mat_b.size(1))
    else:
        size = (mat_a.size(0), mat_a.size(1), mat_b.size(-1))
    return mat_a.new_empty(size, dtype=out_dtype or torch.bfloat16)


_lib: Optional[torch.library.Library] = None


def register(register_meta: bool = True) -> None:
    """Install the Python impl on the op's CUDA key. Idempotent."""
    global _lib
    if _lib is not None:
        return
    _lib = torch.library.Library("aten", "IMPL")
    _lib.impl("_scaled_grouped_mm_v2", _scaled_grouped_mm_v2_impl, "CUDA")
    if register_meta:
        _lib.impl("_scaled_grouped_mm_v2", _scaled_grouped_mm_v2_meta, "Meta")


def unregister() -> None:
    """Drop the override and restore the C++ kernel."""
    global _lib
    if _lib is not None:
        _lib._destroy()
        _lib = None
