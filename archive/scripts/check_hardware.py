import torch
import platform
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")

# Check for other backends (e.g. DirectML for Intel/AMD on Windows)
try:
    import torch_directml
    print("torch_directml is available")
    dml = torch_directml.device()
    print(f"DirectML device: {dml}")
except ImportError:
    print("torch_directml not installed")

# NPU check (Intel OpenVINO or similar might be needed, but standard PyTorch doesn't see NPU usually)
print("NPU check: Standard PyTorch does not natively list NPUs as 'cuda' devices.")
