"""Determine v_mfma_scale_f32_32x32x64_f8f6f4 scale-operand semantics.

A and B are all 1.0, K = 64 (one MMA, two MX blocks). With scale_b = 1.0,
C[m, n] = 32 * (scale_a_eff[m, 0] + scale_a_eff[m, 1]), so reading C tells us
exactly which scale the hardware applied to each (row, k-block).
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl.cdna4 import MFMA_Scale

M = N = 32
K = 64


@flyc.kernel
def probe(out: fx.Tensor, a: fx.Tensor, b: fx.Tensor, ra: fx.Tensor, rb: fx.Tensor,
          tiled_mma: fx.TiledMma):
    tid = fx.thread_idx.x
    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    ra_buf = fx.rocdl.make_buffer_tensor(ra, max_size=True)
    rb_buf = fx.rocdl.make_buffer_tensor(rb, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)

    g2r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
    mma_atom = fx.make_mma_atom(MFMA_Scale(32, 32, 64, fx.Float8E4M3FN))
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(g2r, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(g2r, tiled_mma).get_slice(tid)

    gA = fx.flat_divide(a_buf, (M, K))[None, None, 0, 0]
    gB = fx.flat_divide(b_buf, (N, K))[None, None, 0, 0]
    gC = fx.flat_divide(out_buf, (M, N))[None, None, 0, 0]

    frag_A = thr_mma.make_fragment_A(gA)
    frag_B = thr_mma.make_fragment_B(gB)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0.0)
    fx.copy(g2r, thr_copy_A.partition_S(gA), thr_copy_A.retile(frag_A))
    fx.copy(g2r, thr_copy_B.partition_S(gB), thr_copy_B.retile(frag_B))

    fx.gemm(
        mma_atom.set_value({"scale_a": ra_buf[tid], "scale_b": rb_buf[tid]}),
        frag_C, frag_A, frag_B, frag_C,
    )

    c_row = fx.make_view(0, fx.make_layout((M, N), (1, 0)))
    c_col = fx.make_view(0, fx.make_layout((M, N), (0, 1)))
    thr_c_row = thr_mma.partition_C(c_row)
    thr_c_col = thr_mma.partition_C(c_col)
    for i in range_constexpr(fx.size(frag_C.shape).unpack()):
        gC[fx.get_scalar(thr_c_row[i]), fx.get_scalar(thr_c_col[i])] = frag_C[i]


@flyc.jit
def run(out: fx.Tensor, a: fx.Tensor, b: fx.Tensor, ra: fx.Tensor, rb: fx.Tensor,
        stream: fx.Stream = fx.Stream(None)):
    mma_atom = fx.make_mma_atom(MFMA_Scale(32, 32, 64, fx.Float8E4M3FN))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0)))
    probe._known_block_size = [64, 1, 1]
    probe(out, a, b, ra, rb, tiled_mma).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def run_perm(out: fx.Tensor, a: fx.Tensor, b: fx.Tensor, ra: fx.Tensor, rb: fx.Tensor,
             stream: fx.Stream = fx.Stream(None)):
    mma_atom = fx.make_mma_atom(MFMA_Scale(32, 32, 64, fx.Float8E4M3FN))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((1, 1, 1), (1, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((16, 2, 2), (1, 32, 16))),
    )
    probe._known_block_size = [64, 1, 1]
    probe(out, a, b, ra, rb, tiled_mma).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


a = torch.ones(M, K, device="cuda").to(torch.float8_e4m3fn)
b = torch.ones(N, K, device="cuda").to(torch.float8_e4m3fn)
one = torch.full((64,), 127, dtype=torch.int32, device="cuda")


def case(tag, ra, rb=None):
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    run(out, a, b, ra, rb if rb is not None else one,
        stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    print(f"{tag:44s} C[0,0]={out[0,0].item():8.1f}  C[1,0]={out[1,0].item():8.1f}")
    return out


case("baseline: all bytes = 127 (1.0)", one)

# 2.0 in byte 0 only, on every lane
case("byte0=128 all lanes", torch.full((64,), 128, dtype=torch.int32, device="cuda"))

# 2.0 in byte 1 only, on every lane
case("byte1=128, byte0=127, all lanes",
     torch.full((64,), 127 | (128 << 8), dtype=torch.int32, device="cuda"))

# 2.0 in byte 0, but only on lanes 0-31 (which own k-block 0)
ra = one.clone(); ra[:32] = 128
case("byte0=128 on lanes 0-31 only", ra)

ra = one.clone(); ra[32:] = 128
case("byte0=128 on lanes 32-63 only", ra)

case("byte2=128", torch.full((64,), 127 | (128 << 16), dtype=torch.int32, device="cuda"))

# Asymmetric data: only k-block 0 contributes, so C reveals which lanes' scale
# the hardware applied to block 0.
print("\nA zeroed on k-block 1, so only block 0 contributes (baseline C = 32):")
a[:, 32:] = 0
case("  all 127", one)
ra = one.clone(); ra[:32] = 128
case("  byte0=128 on lanes 0-31 (own block 0)", ra)
ra = one.clone(); ra[32:] = 128
case("  byte0=128 on lanes 32-63 (own block 1)", ra)
