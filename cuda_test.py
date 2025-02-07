import torch
import sys
import subprocess

# Pythonのバージョンを表示
print("Python Version:", sys.version)

# PyTorchのバージョンを表示
print("PyTorch Version:", torch.__version__)

# CUDAのバージョンを表示
print("CUDA Version:", torch.version.cuda)

# CUDAが利用可能かどうかを確認
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# cuDNNのバージョンを表示
print("cuDNN Version:", torch.backends.cudnn.version())

# 利用可能なGPUの数を表示
gpu_count = torch.cuda.device_count()
print("Number of GPUs Available:", gpu_count)

# GPUの情報を表示
if cuda_available and gpu_count > 0:
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
        print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
        print(f"  - Memory Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
else:
    print("No GPUs available or CUDA is not enabled")

# デバイスを設定
device = torch.device("cuda" if cuda_available else "cpu")
print("Using device:", device)

# テスト: Tensorを作成してGPUでの動作を確認
try:
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print("Tensor created on device:", x.device)
    print("Tensor values:", x)
except Exception as e:
    print("Error when creating tensor on device:", e)

# CUDAドライバとツールキットのバージョンを表示
try:
    driver_version = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode("utf-8").strip()
    print("CUDA Driver Version:", driver_version)
except Exception as e:
    print("Error getting CUDA Driver Version:", e)

try:
    toolkit_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    print("CUDA Toolkit Version:", toolkit_version)
except Exception as e:
    print("Error getting CUDA Toolkit Version:", e)
