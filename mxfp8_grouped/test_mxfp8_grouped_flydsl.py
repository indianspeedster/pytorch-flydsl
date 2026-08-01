"""Tests for the _scaled_grouped_mm_v2 MXFP8 override.

The 2d-3d golden comes from per-group torch._scaled_mm, which uses the real
MXFP8 MFMA path on gfx950, so it is independent of the reference dequant.
"""

import os
import sys

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp8_grouped_flydsl as mxg  # noqa: E402


def rand_fp8(*shape, device="cuda"):
    return torch.randn(shape, device=device).to(torch.float8_e4m3fn)


def rand_e8m0(*shape, device="cuda"):
    # e8m0 is a bare biased exponent; keep it near 2^0 and away from 255 (NaN).
    bits = torch.randint(124, 131, shape, dtype=torch.uint8, device=device)
    return bits.view(torch.float8_e8m0fnu)


class TestMXFP8GroupedOverride(TestCase):
    def setUp(self):
        mxg.register_kernel(None)
        mxg.register()

    def tearDown(self):
        mxg.unregister()

    def _run_2d3d(self, g=4, m_per=128, k=256, n=256):
        total_m = g * m_per
        a = rand_fp8(total_m, k)
        b = torch.randn(g, n, k, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
        blocked_k, blocked_n = mxg._round_up(k // 32, 4), mxg._round_up(n, 128)
        sa = rand_e8m0(total_m, blocked_k)
        sb_blocks = rand_e8m0(g, blocked_n, blocked_k)
        offs = torch.arange(m_per, total_m + 1, m_per, dtype=torch.int32, device="cuda")

        out = torch._scaled_grouped_mm_v2(
            a, b, [sa], [mxg.BLOCKWISE_1X32], [mxg.NO_SWIZZLE],
            [sb_blocks.reshape(g, -1)], [mxg.BLOCKWISE_1X32], [mxg.NO_SWIZZLE],
            offs=offs, out_dtype=torch.bfloat16,
        )

        golden = torch.empty_like(out)
        for i in range(g):
            lo, hi = i * m_per, (i + 1) * m_per
            golden[lo:hi] = torch._scaled_mm(
                a[lo:hi], b[i], sa[lo:hi, : k // 32].contiguous(),
                sb_blocks[i, :n, : k // 32].contiguous(), out_dtype=torch.bfloat16,
            )
        return out, golden

    def test_2d3d_matches_scaled_mm(self):
        out, golden = self._run_2d3d()
        self.assertEqual(out.shape, golden.shape)
        torch.testing.assert_close(out.float(), golden.float(), rtol=2e-2, atol=2e-2)

    def test_kernel_hook_is_used(self):
        calls = []

        def fake_kernel(a, b, sa, sb, offs, out, p):
            calls.append(p)
            mxg._reference(a, b, sa, sb, offs, out, p)

        mxg.register_kernel(fake_kernel)
        out, golden = self._run_2d3d()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].case, "2d3d")
        self.assertEqual((calls[0].g, calls[0].m, calls[0].n, calls[0].k), (4, 512, 256, 256))
        torch.testing.assert_close(out.float(), golden.float(), rtol=2e-2, atol=2e-2)

    def test_flydsl_kernel_end_to_end(self):
        import flydsl_grouped

        mxg.register_kernel(flydsl_grouped.kernel)
        out, golden = self._run_2d3d()
        torch.testing.assert_close(out.float(), golden.float(), rtol=3e-2, atol=3e-2)

    def test_flydsl_kernel_ragged_groups(self):
        """Group sizes that are not multiples of the 32-row MMA tile."""
        import flydsl_grouped

        mxg.register_kernel(flydsl_grouped.kernel)
        g, k, n = 3, 256, 256
        sizes = [40, 0, 88]
        total_m = sum(sizes)
        a = rand_fp8(total_m, k)
        b = torch.randn(g, n, k, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
        blocked_k, blocked_n = mxg._round_up(k // 32, 4), mxg._round_up(n, 128)
        sa = rand_e8m0(mxg._round_up(total_m, 128), blocked_k)
        sb_blocks = rand_e8m0(g, blocked_n, blocked_k)
        offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0).to(torch.int32)

        args = (a, b, [sa], [mxg.BLOCKWISE_1X32], [mxg.NO_SWIZZLE],
                [sb_blocks.reshape(g, -1)], [mxg.BLOCKWISE_1X32], [mxg.NO_SWIZZLE])
        out = torch._scaled_grouped_mm_v2(*args, offs=offs, out_dtype=torch.bfloat16)

        mxg.register_kernel(None)  # dequant reference
        ref = torch._scaled_grouped_mm_v2(*args, offs=offs, out_dtype=torch.bfloat16)
        torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2)

    def test_rowwise_falls_through_bitwise(self):
        g, m, n, k = 4, 64, 128, 256
        a = rand_fp8(g * m, k)
        b = torch.randn(g, n, k, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
        sa = torch.rand(g * m, device="cuda") + 0.5
        sb = torch.rand(g, n, device="cuda") + 0.5
        offs = torch.arange(m, g * m + 1, m, dtype=torch.int32, device="cuda")

        args = (a, b, [sa], [mxg.ROWWISE], [mxg.NO_SWIZZLE], [sb], [mxg.ROWWISE], [mxg.NO_SWIZZLE])
        overridden = torch._scaled_grouped_mm_v2(*args, offs=offs, out_dtype=torch.bfloat16)
        mxg.unregister()
        native = torch._scaled_grouped_mm_v2(*args, offs=offs, out_dtype=torch.bfloat16)
        mxg.register()
        self.assertEqual(overridden, native, atol=0, rtol=0)

    def test_unregister_restores_not_implemented(self):
        mxg.unregister()
        with self.assertRaisesRegex(RuntimeError, "requires compile with USE_MSLK"):
            self._run_2d3d()
        mxg.register()

    def test_meta_allows_tracing(self):
        def f(a, b, sa, sb, offs):
            return torch._scaled_grouped_mm_v2(
                a, b, [sa], [mxg.BLOCKWISE_1X32], [mxg.NO_SWIZZLE],
                [sb], [mxg.BLOCKWISE_1X32], [mxg.NO_SWIZZLE],
                offs=offs, out_dtype=torch.bfloat16,
            )

        g, m_per, k, n = 4, 128, 256, 256
        total_m = g * m_per
        with torch.device("meta"):
            a = torch.empty(total_m, k, dtype=torch.float8_e4m3fn)
            b = torch.empty(g, n, k, dtype=torch.float8_e4m3fn).transpose(1, 2)
            sa = torch.empty(total_m, 8, dtype=torch.float8_e8m0fnu)
            sb = torch.empty(g, 256 * 8, dtype=torch.float8_e8m0fnu)
            offs = torch.empty(g, dtype=torch.int32)
            self.assertEqual(f(a, b, sa, sb, offs).shape, torch.Size([total_m, n]))


if __name__ == "__main__":
    run_tests()
