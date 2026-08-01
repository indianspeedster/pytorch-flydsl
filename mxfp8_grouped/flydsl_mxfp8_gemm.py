"""A basic FlyDSL MXFP8 GEMM for gfx950, used as the grouped-GEMM building block.

Computes C[M, N] = (A * sa) @ (B * sb)^T where A is (M, K) and B is (N, K), both
float8_e4m3fn, with e8m0 block scales at 1x32 granularity. Uses the CDNA4 scaled
MFMA atom (v_mfma_scale_f32_32x32x64_f8f6f4), so the scales are applied by the
MMA hardware rather than by a separate dequantize.

Deliberately simple: no LDS staging, no software pipelining, no block swizzle.
It exists so the _scaled_grouped_mm_v2 integration is exercisable end to end;
replace it with an optimized kernel via mxfp8_grouped_flydsl.register_kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl.cdna4 import MFMA_Scale


MMA_M = 32
MMA_N = 32
MMA_K = 64
MX_BLOCK = 32
WAVE_SIZE = 64

# One wave, one MMA tile per block. BLOCK_K == MMA_K keeps exactly one scale
# block per lane per MMA, which is what the atom's i32 scale state holds.
BLOCK_M = MMA_M
BLOCK_N = MMA_N
BLOCK_K = MMA_K
BLOCK_THREADS = WAVE_SIZE
MX_BLOCKS_PER_MMA = BLOCK_K // MX_BLOCK
_HALF_BLOCK = MX_BLOCK // 2


def _k_permutation():
    """Reorder the MMA's K mode so each lane owns one whole MX block.

    Without it, lane group g holds logical k in
    [16g, 16g+16) union [32+16g, 32+16g+16) -- half of each MX block -- so a
    single e8m0 scale per lane cannot express per-block scaling. Measured with
    probe_kmap.py; the permutation maps (v, g, u) -> v + 32g + 16u so that
    group g ends up owning exactly block g.
    """
    return fx.make_layout(
        (_HALF_BLOCK, MX_BLOCKS_PER_MMA, MX_BLOCKS_PER_MMA),
        (1, MX_BLOCK, _HALF_BLOCK),
    )


@flyc.kernel
def mxfp8_gemm_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    sa: fx.Tensor,
    sb: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    kb: fx.Int32,
    tiled_mma: fx.TiledMma,
):
    tid = fx.thread_idx.x
    num_pid_n = (n + BLOCK_N - 1) // BLOCK_N
    bid_m = fx.block_idx.x // num_pid_n
    bid_n = fx.block_idx.x % num_pid_n

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    sa_buf = fx.rocdl.make_buffer_tensor(sa, max_size=True)
    sb_buf = fx.rocdl.make_buffer_tensor(sb, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)

    g2r_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
    # Rebuilt here (rather than reusing the host-side atom inside tiled_mma)
    # because the per-lane scales are atom state, and only MmaAtom carries it.
    mma_atom = fx.make_mma_atom(MFMA_Scale(MMA_M, MMA_N, MMA_K, fx.Float8E4M3FN))

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(g2r_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(g2r_copy_atom, tiled_mma).get_slice(tid)

    gC = fx.flat_divide(out_buf, (BLOCK_M, BLOCK_N))[None, None, bid_m, bid_n]
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0.0)

    # Coordinate views let flydsl tell us which row each lane owns, instead of
    # hardcoding the MFMA register layout.
    ab_row = fx.make_view(0, fx.make_layout((BLOCK_M, BLOCK_K), (1, 0)))
    thr_a_row = thr_mma.partition_A(ab_row)
    thr_b_row = thr_mma.partition_B(ab_row)

    c_row = fx.make_view(0, fx.make_layout((BLOCK_M, BLOCK_N), (1, 0)))
    c_col = fx.make_view(0, fx.make_layout((BLOCK_M, BLOCK_N), (0, 1)))
    thr_c_row = thr_mma.partition_C(c_row)
    thr_c_col = thr_mma.partition_C(c_col)

    a_m = fx.get_scalar(thr_a_row[0]) + bid_m * BLOCK_M
    b_n = fx.get_scalar(thr_b_row[0]) + bid_n * BLOCK_N
    # After the K permutation, lane group g owns MX block g of the MMA's K span
    # (measured by probe_kmap.py), and that is the block its scale byte governs.
    mx_slot = (tid % WAVE_SIZE) // (WAVE_SIZE // MX_BLOCKS_PER_MMA)

    tiled_A = fx.flat_divide(a_buf, (BLOCK_M, BLOCK_K))
    tiled_B = fx.flat_divide(b_buf, (BLOCK_N, BLOCK_K))
    frag_A = thr_mma.make_fragment_A(tiled_A[None, None, bid_m, 0])
    frag_B = thr_mma.make_fragment_B(tiled_B[None, None, bid_n, 0])
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)

    # The loop body lives in a nested function so that the DSL's scf.for
    # rewriter does not try to carry the thread-sliced mma/copy objects as
    # loop-carried values.
    def k_step(k_tile):
        gA = tiled_A[None, None, bid_m, k_tile]
        gB = tiled_B[None, None, bid_n, k_tile]
        fx.copy(g2r_copy_atom, thr_copy_A.partition_S(gA), frag_A_retile)
        fx.copy(g2r_copy_atom, thr_copy_B.partition_S(gB), frag_B_retile)

        # With opsel 0 the atom reads byte 0 of each lane's i32 scale state and
        # applies it to the MX block that lane owns (measured by probe_scale.py).
        k_blk = k_tile * MX_BLOCKS_PER_MMA + mx_slot
        sa_val = sa_buf[a_m * kb + k_blk].to(fx.Int32)
        sb_val = sb_buf[b_n * kb + k_blk].to(fx.Int32)

        fx.gemm(
            mma_atom.set_value({"scale_a": sa_val, "scale_b": sb_val}),
            frag_C,
            frag_A,
            frag_B,
            frag_C,
        )

    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    for k_tile in range(0, k_tiles, 1):
        k_step(k_tile)

    for i in range_constexpr(fx.size(frag_C.shape).unpack()):
        row = fx.get_scalar(thr_c_row[i])
        col = fx.get_scalar(thr_c_col[i])
        if (bid_m * BLOCK_M + row < m) & (bid_n * BLOCK_N + col < n):
            gC[row, col] = frag_C[i].to(fx.BFloat16)


@flyc.jit
def mxfp8_gemm(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    sa: fx.Tensor,
    sb: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    kb: fx.Int32,
    grid_x: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    mma_atom = fx.make_mma_atom(MFMA_Scale(MMA_M, MMA_N, MMA_K, fx.Float8E4M3FN))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((1, 1, 1), (1, 1, 0)),
        fx.make_tile(None, None, _k_permutation()),
    )
    mxfp8_gemm_kernel._known_block_size = [BLOCK_THREADS, 1, 1]
    mxfp8_gemm_kernel(out, a, b, sa, sb, m, n, k, kb, tiled_mma).launch(
        grid=(grid_x, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def supports(k: int) -> bool:
    """K must tile exactly: a partial K tile would read into the next row rather
    than off the end of the buffer, so bounds checking would not catch it."""
    return k % BLOCK_K == 0
