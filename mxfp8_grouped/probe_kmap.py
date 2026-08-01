"""Map each logical k index to the lane group whose scale byte governs it.

A has a single nonzero k column, B is all ones, scale_b = 1.0. Lanes 0-31 get
scale 2.0 and lanes 32-63 get 1.0, so C[0,0] is 2.0 exactly when that logical k
is covered by the lanes 0-31 scale.
"""

import torch
import flydsl.expr as fx

import os
from probe_scale import run, run_perm, M, N, K  # reuses the probe kernel

launch = run_perm if os.environ.get('PERM') else run

lo = torch.full((64,), 128, dtype=torch.int32, device="cuda")
lo[32:] = 127
one = torch.full((64,), 127, dtype=torch.int32, device="cuda")

b = torch.ones(N, K, device="cuda").to(torch.float8_e4m3fn)
owner = []
for k in range(K):
    a = torch.zeros(M, K, device="cuda")
    a[:, k] = 1.0
    a = a.to(torch.float8_e4m3fn)
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    launch(out, a, b, lo, one, stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    owner.append(0 if abs(out[0, 0].item() - 2.0) < 1e-3 else 1)

print("logical k -> lane group owning its scale (0 = lanes 0-31, 1 = lanes 32-63):")
for start in range(0, K, 16):
    print(f"  k[{start:2d}:{start + 16:2d}] {''.join(str(v) for v in owner[start:start + 16])}")

runs = []
cur, n = owner[0], 0
for v in owner:
    if v == cur:
        n += 1
    else:
        runs.append((cur, n))
        cur, n = v, 1
runs.append((cur, n))
print("runs:", runs)
