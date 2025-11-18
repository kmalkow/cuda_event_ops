from setuptools import find_packages, setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --------------------------------------------------------------------
# 1. Determine host C++ compiler for nvcc
# --------------------------------------------------------------------
# Priority:
#   1) CUDAHOSTCXX
#   2) CXX
#   3) fallback to "g++"
host_cxx = os.environ.get("CUDAHOSTCXX") or os.environ.get("CXX") or "g++"

print(f"[cuda_event_ops] Using host C++ compiler for nvcc: {host_cxx}")

# --------------------------------------------------------------------
# 2. Match PyTorch's C++11 ABI setting
# --------------------------------------------------------------------
# PyTorch exposes this value so extensions use a compatible ABI.
# It is a bool (True/False), so convert to int (1/0).
cxx11_abi = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))
print(f"[cuda_event_ops] Using _GLIBCXX_USE_CXX11_ABI={cxx11_abi}")

cxx_args = [
    "-O3",
    f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
]

nvcc_args = [
    "-O3",
    f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
    f"-ccbin={host_cxx}",
]

# --------------------------------------------------------------------
# 3. Set CUDA arch list if no GPU is visible during build
# --------------------------------------------------------------------
if not torch.cuda.is_available():
    if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
        # turing, ampere, ada
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
    install_requires=["torch"],
)
