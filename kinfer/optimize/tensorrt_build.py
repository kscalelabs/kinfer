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

def check_tensorrt_compatibility() -> bool:
    """Check if system is compatible with TensorRT."""
    system = platform.system().lower()

    # TensorRT only supports Linux and Windows
    if system not in ("linux", "windows"):
        return False

    # Check for CUDA
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
