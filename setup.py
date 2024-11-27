#!/usr/bin/env python
"""Setup script for kinfer package."""

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.install import install


def is_tegra_platform() -> bool:
    """Check if running on a Tegra platform."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return any(platform in model for platform in ["tegra", "jetson"])
    except FileNotFoundError:
        return False

PLATFORM = platform.system().lower()
MACHINE = platform.machine().lower()

# Define platform-specific TensorRT requirements
if PLATFORM in ("linux", "windows"):
    if MACHINE == "aarch64" and is_tegra_platform():
        # On Tegra platforms, TensorRT should be installed via apt
        tensorrt_requires = []
    else:
        tensorrt_requires = [
            'tensorrt>=10.0.0',
            'cuda-python>=12.0.0',
            'onnx-tensorrt'
        ]
else:
    tensorrt_requires = []

class CustomBuildCommand(build_py):
    """Custom build command to handle onnx-tensorrt."""
    def run(self) -> None:
        # Only build TensorRT support on supported platforms
        if PLATFORM in ("linux", "windows"):
            if MACHINE == "aarch64" and is_tegra_platform():
                print("INFO: On Tegra platform, please install TensorRT using:")
                print("      sudo apt-get install tensorrt")
                return super().run()
                
            onnx_tensorrt_path = Path(__file__).parent / "third_party/onnx-tensorrt"
            
            # Remove existing directory if it exists
            if onnx_tensorrt_path.exists():
                import shutil
                shutil.rmtree(onnx_tensorrt_path)
            
            # Clone with submodules
            subprocess.run([
                "git", "clone", 
                "--recursive",
                "https://github.com/onnx/onnx-tensorrt.git",
                str(onnx_tensorrt_path)
            ], check=True)
            
            # Initialize and update submodules
            subprocess.run([
                "git", "submodule", "update", "--init", "--recursive"
            ], cwd=onnx_tensorrt_path, check=True)

            # Build onnx-tensorrt
            build_dir = onnx_tensorrt_path / "build"
            build_dir.mkdir(exist_ok=True)
            
            # Find TensorRT Python include directory
            import site
            possible_paths = [
                # Common paths for TensorRT Python package
                "/usr/local/lib/python3/dist-packages/tensorrt/include",
                "/usr/lib/python3/dist-packages/tensorrt/include",
                # Try to find in site-packages
                *[f"{p}/tensorrt/include" for p in site.getsitepackages()],
                # Try conda env
                os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib/python3/dist-packages/tensorrt/include"),
            ]
            
            tensorrt_python_include = None
            for path in possible_paths:
                if Path(path).exists():
                    tensorrt_python_include = path
                    break
                
            if not tensorrt_python_include:
                print("WARNING: Could not find TensorRT Python include directory")
                print("Searched paths:", possible_paths)
                return super().run()
            
            print(f"Found TensorRT Python include directory: {tensorrt_python_include}")
            
            subprocess.run([
                "cmake", "..",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DTENSORRT_PYTHON_INCLUDE_DIR={tensorrt_python_include}"
            ], cwd=build_dir, check=True)

            subprocess.run([
                "cmake", "--build", ".", 
                "--config", "Release",
                "-j"
            ], cwd=build_dir, check=True)

            # Create onnx_tensorrt package directory
            package_dir = Path(self.build_lib) / "onnx_tensorrt"
            package_dir.mkdir(exist_ok=True)
            
            # Copy the built module
            built_module = build_dir / "onnx_tensorrt.so"  # or .pyd on Windows
            if built_module.exists():
                import shutil
                shutil.copy2(built_module, package_dir / built_module.name)
                
                # Create __init__.py
                with open(package_dir / "__init__.py", "w") as f:
                    f.write("from .onnx_tensorrt import *\n")
            else:
                print(f"WARNING: Built module not found at {built_module}")

        super().run()

setup(
    name="kinfer",
    version="0.0.1",
    description="Neural network inference toolkit",
    author="K-Scale Labs",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "onnx",
        "onnxruntime",
    ],
    extras_require={
        'tensorrt': tensorrt_requires,
    },
    cmdclass={
        'build_py': CustomBuildCommand,
    },
)
