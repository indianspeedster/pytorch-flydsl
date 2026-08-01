"""Grouped MXFP8 GEMM built on the basic FlyDSL kernel.

Drives one kernel launch per group. Group boundaries live on the device in
``offs``, so this costs a d2h sync per call -- acceptable for a placeholder, and
the same sync the dequant reference already pays. An optimized replacement would
launch a single kernel that reads ``offs`` on device.

Register it with::

    import mxfp8_grouped_flydsl as mxg, flydsl_grouped
    mxg.register_kernel(flydsl_grouped.kernel)
    mxg.register()
"""

import torch

from flydsl_mxfp8_gemm import BLOCK_M, BLOCK_N, mxfp8_gemm, supports
import flydsl.expr as fx


def _launch(a, b, sa, sb, out):
    """a (M, K) fp8, b (N, K) fp8, sa (M, kb) / sb (N, kb) e8m0, out (M, N) bf16."""
    m, k = a.shape
    n = b.shape[0]
    kb = sa.shape[1]
    grid = ((m + BLOCK_M - 1) // BLOCK_M) * ((n + BLOCK_N - 1) // BLOCK_N)
    mxfp8_gemm(
        out, a, b,
        sa.reshape(-1), sb.reshape(-1),
        m, n, k, kb, grid,
        stream=fx.Stream(torch.cuda.current_stream().cuda_stream),
    )


def kernel(mat_a, mat_b, scale_a, scale_b, offs, out, p):
    bounds = offs.tolist()
    # In the 2d-2d (wgrad) form the contraction dim is the ragged one, so every
    # group length must tile exactly -- not just the total.
    contractions = [p.k] if p.case == "2d3d" else [
        hi - lo for lo, hi in zip([0] + bounds[:-1], bounds)
    ]
    bad = [c for c in contractions if c and not supports(c)]
    if bad:
        raise NotImplementedError(
            "basic FlyDSL MXFP8 kernel requires every contraction length to be a "
            f"multiple of 64, got {sorted(set(bad))}"
        )

    kb = p.k // 32
    sa_all = scale_a.view(torch.uint8)
    sb_all = scale_b.view(torch.uint8)
    start = 0
    scol = 0  # ragged-K scales are padded per group, so track the column offset
    for g, end in enumerate(bounds):
        if end <= start:
            start = end
            continue
        if p.case == "2d3d":
            a_g = mat_a[start:end]
            b_g = mat_b[g].t().contiguous()
            sa_g = sa_all[start:end, :kb].contiguous()
            sb_g = sb_all[g].view(p.blocked_n, p.blocked_k)[: p.n, :kb].contiguous()
            _launch(a_g, b_g, sa_g, sb_g, out[start:end])
        else:
            kb_g = (end - start) // 32
            a_g = mat_a[:, start:end].contiguous()
            b_g = mat_b[start:end].t().contiguous()
            sa_g = sa_all[: p.m, scol : scol + kb_g].contiguous()
            sb_g = sb_all[: p.n, scol : scol + kb_g].contiguous()
            _launch(a_g, b_g, sa_g, sb_g, out[g])
            scol += ((kb_g + 3) // 4) * 4
        start = end
