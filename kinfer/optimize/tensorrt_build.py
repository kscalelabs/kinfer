"""TensorRT build configuration and checks."""
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


def get_tensorrt_path() -> Optional[str]:
    """Get TensorRT installation path."""
    # Common TensorRT installation paths
    possible_paths = [
        "/usr/local/tensorrt",
        "/usr/lib/tensorrt",
        os.environ.get("TENSORRT_PATH"),
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            return path
    return None

def is_tegra_platform() -> bool:
    """Check if running on a Tegra platform."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return any(platform in model for platform in ["tegra", "jetson"])
    except FileNotFoundError:
        return False

def check_tensorrt_compatibility() -> bool:
    """Check if system is compatible with TensorRT."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # TensorRT only supports Linux and Windows
    if system not in ("linux", "windows"):
        return False

    # Both x86_64 and aarch64 are supported
    supported_architectures = ("x86_64", "amd64", "aarch64", "arm64")
    if machine not in supported_architectures:
        return False

    # For Tegra platforms, check if TensorRT is installed via apt
    if system == "linux" and is_tegra_platform():
        try:
            subprocess.run(
                ["dpkg", "-l", "tensorrt"], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("TensorRT not found. Please install using: sudo apt-get install tensorrt")
            return False

    # For non-Tegra platforms, check for CUDA
    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    return True

def build_tensorrt_parser() -> None:
    """Build TensorRT parser from source."""
    if not check_tensorrt_compatibility():
        print("System is not compatible with TensorRT. Skipping build.")
        return

    build_path = Path("third_party/onnx-tensorrt/build")
    build_path.mkdir(exist_ok=True, parents=True)

    subprocess.run(
        [
            "cmake", "..",
            f"-DTENSORRT_ROOT={get_tensorrt_path() or '/usr/local/tensorrt'}",
        ],
        cwd=build_path,
        check=True,
    )

    subprocess.run(["make", "-j"], cwd=build_path, check=True)
    subprocess.run(["python", "setup.py", "install"], cwd="third_party/onnx-tensorrt", check=True)
