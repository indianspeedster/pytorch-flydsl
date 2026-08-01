# MXFP8 grouped GEMM via FlyDSL (ROCm gfx950)

On ROCm the MXFP8 arm of `_scaled_grouped_mm_v2` is compiled out --
`aten/src/ATen/native/hip/GroupedBlas.cpp:171` guards the MSLK call with
`!defined(USE_ROCM)` -- so every `BlockWise1x32` call raises
`NotImplementedError`. MSLK's `mx8mx8bf16_grouped` kernels are CUTLASS-only; its
ROCm arm builds only `fp8_rowwise_grouped`.

FlyDSL kernels JIT from Python, so that C++ arm cannot launch one. This replaces
the op's CUDA kernel from Python instead, which needs no PyTorch rebuild and
keeps existing MoE code working unchanged.

## Use

```python
import mxfp8_grouped_flydsl as mxg
import flydsl_grouped

mxg.register_kernel(flydsl_grouped.kernel)
mxg.register()
```

Non-MXFP8 recipes forward to `aten::_scaled_grouped_mm` (v1), verified bitwise
identical to the un-overridden path. Leaving the kernel unset falls back to a
dequantize-and-matmul reference.

## Files

| file | what |
| --- | --- |
| `mxfp8_grouped_flydsl.py` | dispatch shim on the op's CUDA key, validation, reference impl, meta kernel |
| `flydsl_mxfp8_gemm.py` | the FlyDSL kernel, using the CDNA4 scaled MFMA atom |
| `flydsl_grouped.py` | grouped driver, one launch per group |
| `test_mxfp8_grouped_flydsl.py` | test suite; 2d-3d numerics golden is per-group `torch._scaled_mm` |
| `probe_*.py` | measurements the kernel's correctness rests on (see below) |
| `debug_kernel.py` | correctness matrix against a dequant golden |

## Measured hardware semantics

FlyDSL ships no MXFP8 examples, so `v_mfma_scale_f32_32x32x64_f8f6f4` semantics
were determined empirically:

* With `opsel = 0` only **byte 0** of the atom's i32 scale state is read
  (`probe_scale.py`). Bytes 1-3 are inert.
* Without a K permutation, lane group `g` holds logical
  `k in [16g, 16g+16) union [32+16g, 32+16g+16)` -- half of *each* MX block --
  so one scale byte per lane cannot express per-block scaling at all
  (`probe_kmap.py`). Passing
  `make_tile(None, None, make_layout((16,2,2), (1,32,16)))` to `make_tiled_mma`
  regroups it so lane group `g` owns block `g`. This is invisible in unscaled
  GEMM, since a consistent k-permutation cancels between A and B.
* `partition_A`'s coordinate view does *not* reflect that permutation; the lane
  group is the reliable scale index.

## Status

The kernel is deliberately basic: no LDS staging, no software pipelining, no
block swizzle, one wave per block, 32x32x64 tiles. It exists so the integration
is exercisable end to end, not to be competitive. Measured 53.9 TFLOP/s against
43.3 for the dequant reference (G=8, M=512/group, K=N=2048) -- far off what
gfx950 can do on MXFP8.

Coverage: forward and dgrad (both 2d-3d) work. wgrad (2d-2d, contraction over
the ragged dim) works only when every group's length is a multiple of 64; the
kernel rejects other cases rather than reading out of bounds. 32 is a floor
imposed by MXFP8 itself, so wgrad needs per-expert token counts padded to a
multiple of 32 regardless of kernel quality.
