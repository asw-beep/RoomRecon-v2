# GPU machine environment

Machine: GPU machine, NVIDIA A100-PCIE-40GB, MIG 7g.40gb (whole card), shared with another
tenant; driver 555.42.06; Ubuntu 20.04 Docker container. Set up 2026-10-03 with
`scripts/setup_gpu.sh` (no sudo).

| Component | Version (printed by the setup / checks) |
|---|---|
| pixi | 0.81.0 |
| DN-Splatter | `97588b4290128ce7ba6fdbfaac3020b42b17de4c` (`env/dn-splatter.commit`) |
| Python | 3.10 (pixi) |
| torch | 2.2.2, CUDA 11.8, `torch.cuda.is_available()` True, device "NVIDIA A100-PCIE-40GB MIG 7g.40gb" |
| gsplat | 1.0.0 |
| numpy | 1.26.4 |
| nerfstudio | 1.1.3 (PyPI; see deviation 1) |

Lockfile: `env/pixi.lock` (copied from the GPU machine after the successful install).

Deviations from DN-Splatter's own `pixi.toml` (applied by `setup_gpu.sh`):
1. `nerfstudio` from PyPI `==1.1.3` instead of git commit `a64026f`: upstream's manifest
   conflicts with its own `pyproject.toml` pin (`nerfstudio == 1.1.3`), so pixi could not solve.
2. `numpy >=1.26,<2`, `cython <3.1`, `setuptools` added, and `pymcubes` built without build
   isolation: `pymcubes==0.1.2` failed to compile against NumPy 2 (`PyArray_DOUBLE` removed).

Free GPU memory at the first check: `free_mib=5991` (MIG row 34456 / 40447 MiB used by the
other tenant), 2026-10-03.
