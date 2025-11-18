from setuptools import find_packages, setup
import os

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --------------------------------------------------------------------
# 1. Determine host C++ compiler for nvcc
# --------------------------------------------------------------------
# Priority:
#   1) CUDAHOSTCXX (if set)
#   2) CXX (if set)
#   3) fallback to "g++" on PATH
host_cxx = os.environ.get("CUDAHOSTCXX") or os.environ.get("CXX") or "g++"
print(f"[cuda_event_ops] Using host C++ compiler for nvcc: {host_cxx}")

# C++ compiler flags
cxx_args = [
    "-O3",
]

# nvcc flags: optimization + explicit host compiler
nvcc_args = [
    "-O3",
    f"-ccbin={host_cxx}",
]

# --------------------------------------------------------------------
# 2. Set CUDA arch list if no GPU is visible during build
# --------------------------------------------------------------------
if not torch.cuda.is_available():
    if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
        # Turing, Ampere, Ada
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.7;8.9+PTX"


setup(
    name="cuda_event_ops",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="iterative_3d_warp_cuda._C",
            sources=[
                "cuda_event_ops/iterative_3d_warp/extension.cpp",
                "cuda_event_ops/iterative_3d_warp/kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
        CUDAExtension(
            name="trilinear_splat_cuda._C",
            sources=[
                "cuda_event_ops/trilinear_splat/extension.cpp",
                "cuda_event_ops/trilinear_splat/kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[],
)
