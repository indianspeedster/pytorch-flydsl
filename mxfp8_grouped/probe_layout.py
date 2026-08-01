"""Dump the (m, k) coordinates each lane owns in the MFMA_Scale A fragment."""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl.cdna4 import MFMA_Scale

MMA_M, MMA_N, MMA_K = 32, 32, 64


@flyc.kernel
def probe(rows: fx.Tensor, cols: fx.Tensor, nvals: fx.Tensor, tiled_mma: fx.TiledMma):
    tid = fx.thread_idx.x
    rows_buf = fx.rocdl.make_buffer_tensor(rows, max_size=True)
    cols_buf = fx.rocdl.make_buffer_tensor(cols, max_size=True)
    nvals_buf = fx.rocdl.make_buffer_tensor(nvals, max_size=True)
    thr_mma = tiled_mma.thr_slice(tid)

    ab_row = fx.make_view(0, fx.make_layout((MMA_M, MMA_K), (1, 0)))
    ab_col = fx.make_view(0, fx.make_layout((MMA_M, MMA_K), (0, 1)))
    thr_a_row = thr_mma.partition_A(ab_row)
    thr_a_col = thr_mma.partition_A(ab_col)

    n = fx.size(thr_a_col.shape).unpack()
    nvals_buf[tid] = fx.Int32(n)
    for i in range_constexpr(n):
        rows_buf[tid * 64 + i] = fx.Int32(fx.get_scalar(thr_a_row[i]))
        cols_buf[tid * 64 + i] = fx.Int32(fx.get_scalar(thr_a_col[i]))


@flyc.jit
def run_probe(rows: fx.Tensor, cols: fx.Tensor, nvals: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    mma_atom = fx.make_mma_atom(MFMA_Scale(MMA_M, MMA_N, MMA_K, fx.Float8E4M3FN))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 0)))
    probe._known_block_size = [64, 1, 1]
    probe(rows, cols, nvals, tiled_mma).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


rows = torch.full((64 * 64,), -1, dtype=torch.int32, device="cuda")
cols = torch.full((64 * 64,), -1, dtype=torch.int32, device="cuda")
nvals = torch.zeros(64, dtype=torch.int32, device="cuda")
run_probe(rows, cols, nvals, stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
torch.cuda.synchronize()

n = nvals[0].item()
print("values per lane:", n)
r = rows.view(64, 64)[:, :n].cpu()
c = cols.view(64, 64)[:, :n].cpu()
for lane in (0, 1, 31, 32, 33, 63):
    print(f"lane {lane:2d}: m={sorted(set(r[lane].tolist()))} k={c[lane].tolist()}")
print("k-blocks touched per lane:", sorted({tuple(sorted({v // 32 for v in c[l].tolist()})) for l in range(64)}))
