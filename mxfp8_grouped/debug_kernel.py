import os
import sys

import torch
import flydsl.expr as fx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flydsl_mxfp8_gemm import mxfp8_gemm, BLOCK_M, BLOCK_N  # noqa: E402


def pad(s, rows):
    """Compact (rows, kb) -> the padded layout torch._scaled_mm expects."""
    kb_pad = ((s.shape[1] + 3) // 4) * 4
    rows_pad = ((rows + 127) // 128) * 128
    out = torch.zeros(rows_pad, kb_pad, dtype=torch.uint8, device=s.device)
    out[: s.shape[0], : s.shape[1]] = s
    return out


def run_case(tag, M, N, K, sa, sb):
    kb = K // 32
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)

    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    grid = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    mxfp8_gemm(
        out, a, b, sa.reshape(-1), sb.reshape(-1), M, N, K, kb, grid,
        stream=fx.Stream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    # Dequant golden: unambiguous MXFP8 semantics, and already cross-checked
    # against torch._scaled_mm in the module test suite.
    af = a.float() * sa.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
    bf = b.float() * sb.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
    gold = af @ bf.t()

    err = (out - gold).abs().max().item()
    ok = err < max(1e-2, 3e-2 * gold.abs().max().item())
    print(f"{tag:46s} maxerr={err:9.4f} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("   out :", [round(v, 2) for v in out[0, :6].tolist()])
        print("   gold:", [round(v, 2) for v in gold[0, :6].tolist()])


def by_k(rows, kb):
    return (127 + (torch.arange(kb, device="cuda") % 3)).to(torch.uint8)[None, :].expand(rows, kb).contiguous()


def by_row(rows, kb):
    return (127 + (torch.arange(rows, device="cuda") % 3)).to(torch.uint8)[:, None].expand(rows, kb).contiguous()


def ones(rows, kb):
    return torch.full((rows, kb), 127, dtype=torch.uint8, device="cuda")


# K=64: a single k-tile, so the two MX blocks live in one MFMA.
run_case("K=64  scale_a by k-block", 128, 128, 64, by_k(128, 2), ones(128, 2))
run_case("K=64  scale_a by row", 128, 128, 64, by_row(128, 2), ones(128, 2))
# K=128: two k-tiles, exercising the k_tile term of the scale index.
run_case("K=128 scale_a by k-block", 128, 128, 128, by_k(128, 4), ones(128, 4))
run_case("K=128 scale_a by row", 128, 128, 128, by_row(128, 4), ones(128, 4))
run_case("K=128 both scales random", 128, 128, 128,
         torch.randint(126, 129, (128, 4), dtype=torch.uint8, device="cuda"),
         torch.randint(126, 129, (128, 4), dtype=torch.uint8, device="cuda"))
