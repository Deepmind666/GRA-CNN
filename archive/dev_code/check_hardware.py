import torch
import sys
import os
import platform

def check_hardware():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OS: {platform.system()} {platform.release()}")
    
    print("\n--- CUDA (NVIDIA) ---")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("CUDA Available: No")

    print("\n--- MPS (Apple Silicon) ---")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS Available: Yes")
    else:
        print("MPS Available: No")

    print("\n--- DirectML (Windows/Intel/AMD) ---")
    try:
        import torch_directml
        print("torch-directml installed: Yes")
        dml = torch_directml.device()
        print(f"DirectML Device: {dml}")
        # Try to get device name if possible
        print(f"DirectML Device Count: {torch_directml.device_count()}")
        for i in range(torch_directml.device_count()):
            print(f"Device {i}: {torch_directml.device_name(i)}")
    except ImportError:
        print("torch-directml installed: No")

    print("\n--- Intel Extension for PyTorch (IPEX) ---")
    try:
        import intel_extension_for_pytorch as ipex
        print("IPEX installed: Yes")
        # Check XPU devices
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
             print(f"XPU Available: Yes")
             print(f"XPU Device Count: {torch.xpu.device_count()}")
             for i in range(torch.xpu.device_count()):
                 print(f"Device {i}: {torch.xpu.get_device_name(i)}")
        else:
             print("XPU Available: No")
    except ImportError:
        print("IPEX installed: No")

    print("\n--- OpenVINO ---")
    try:
        from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        print(f"OpenVINO Available Devices: {devices}")
        for device in devices:
            print(f"  {device}: {core.get_property(device, 'FULL_DEVICE_NAME')}")
    except ImportError:
        print("OpenVINO Runtime installed: No")

if __name__ == "__main__":
    check_hardware()
