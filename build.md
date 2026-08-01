# Building this branch from source (ROCm gfx950)

Build runbook for `flydsl-mxfp8-grouped`, which carries PR #190903's FlyDSL
inductor backend plus the MXFP8 grouped GEMM in `mxfp8_grouped/`.

Verified against: ROCm 7.0.2, gfx950 (MI350-class), Python 3.12.3, 192 cores,
with `ccache`, `cmake` and `ninja` on PATH. Adjust `PYTORCH_ROCM_ARCH` and
`MAX_JOBS` for other machines.

Three steps below are failure modes that each cost a full build. They are marked
**GOTCHA**. Do not skip them.

## 1. Clone

```bash
git clone --branch flydsl-mxfp8-grouped \
    https://github.com/indianspeedster/pytorch-flydsl.git
cd pytorch-flydsl
git submodule update --init --recursive
```

## 2. Environment and dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-build.txt
pip install -r requirements.txt
```

`requirements.txt` pins `flydsl==0.3.0.dev765`, so FlyDSL is installed here and
needs no separate step. The pin is load-bearing: released versions lack
`fx.rocdl.get_buffer_rsrc`, which
`torch/_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py` calls,
and `fx.rocdl.cdna4.MFMA_Scale`, which the MXFP8 kernel is built on. It is a
~272 MB ROCm-only wheel.

Note that `torch/_inductor/codegen/flydsl/flydsl_utils.py` only checks that the
package imports and that its runtime `.so` loads -- it never checks a version.
With a released flydsl the backend is still selected and then dies at codegen
rather than being skipped.

## 3. GOTCHA: hipify must run before cmake

```bash
python tools/amd_build/build_amd.py
```

`pip install -e .` does **not** invoke this. Without it cmake fails with
`File c10/hip/impl/hip_cmake_macros.h.in does not exist`.

It rewrites ~148 tracked files in place (`USE_CUDA` -> `USE_ROCM`,
`ATen/cuda/` -> `ATen/hip/`), so `git status` will show a large dirty tree
afterwards. That is expected build state. Never commit those files; when
committing, stage explicit paths rather than using `git commit -a` or
`git add -A`.

## 4. Build

```bash
export USE_ROCM=1 USE_CUDA=0 PYTORCH_ROCM_ARCH=gfx950
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export BUILD_TEST=0 CMAKE_BUILD_TYPE=Release MAX_JOBS=160
export USE_CCACHE=1
export CMAKE_C_COMPILER_LAUNCHER=ccache \
       CMAKE_CXX_COMPILER_LAUNCHER=ccache \
       CMAKE_HIP_COMPILER_LAUNCHER=ccache

# GOTCHA: see below
export CMAKE_ARGS="-DCMAKE_EXE_LINKER_FLAGS='-L/opt/rocm/lib -Wl,-rpath-link,/opt/rocm/lib'"

pip install -e . -v --no-build-isolation
```

`pip install -e . -v --no-build-isolation` is the only supported build command
for this repo (see CLAUDE.md). Do not substitute `setup.py` invocations.

The first full build is long; ccache makes rebuilds substantially cheaper.

### GOTCHA: stale system libhsa breaks linking

`/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1` is v1.11.0 and lacks
`hsa_amd_vmem_*`. cmake puts that directory on `-rpath-link` ahead of ROCm's, so
`torch_shm_manager` fails to link with undefined `ROCR_1` references. The
`CMAKE_ARGS` line above fixes it.

The build backend is scikit-build-core, which honors `CMAKE_ARGS` on a fresh
configure. **Once `build/CMakeCache.txt` exists, environment settings are
ignored.** If you hit this mid-build, edit `CMAKE_EXE_LINKER_FLAGS` in
`build/CMakeCache.txt` directly instead of re-exporting.

## 5. GOTCHA: GPU access needs group membership

`/dev/kfd` is `root:render` mode 660.

```bash
sudo usermod -aG render,video $USER
```

Group membership does not apply until the next login. Until then, wrap every
command that touches the GPU:

```bash
sg render -c "sg video -c 'python your_script.py'"
```

Symptom if missing: no visible devices, or HIP failing to initialize.

## 6. Triton (optional)

Not needed for the MXFP8 kernel, but `torch._dynamo` imports it. The repo pins
3.8.0 while the newest prebuilt `pytorch-triton-rocm` is 3.6.0, so it must be
built from the pinned commit in `.ci/docker/ci_commit_pins/triton.txt`
(`pip install pybind11` first).

Do not put the *parent directory* of a Triton source checkout on `sys.path`. A
bare `triton/` directory without `__init__.py` becomes a namespace package that
shadows the installed one, and the failure surfaces far away as
`AttributeError: module 'triton' has no attribute 'language'` from inside
`torch/_dynamo/utils.py`.

## 7. Verify

```bash
python -c "import torch; print(torch.__version__, torch.version.hip,
    torch.cuda.get_device_properties(0).gcnArchName)"

python mxfp8_grouped/test_mxfp8_grouped_flydsl.py   # expect 7 passed
python mxfp8_grouped/debug_kernel.py                # scale-mapping matrix
```

The test suite is the real end-to-end check: it drives
`torch._scaled_grouped_mm_v2` through the Python override with the FlyDSL kernel
registered, against a per-group `torch._scaled_mm` golden that uses the hardware
MXFP8 MFMA path.

To confirm the FlyDSL inductor backend specifically, autotuning is opt-in behind
`FLYDSL_ENABLE_AUTOTUNING=1`; without it only a single config is tried.
