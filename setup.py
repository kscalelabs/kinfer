#!/usr/bin/env python
"""Setup script for kinfer package."""

import os
import platform
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.install import install

PLATFORM = platform.system().lower()

# Define platform-specific TensorRT requirements
if PLATFORM in ("linux", "windows"):
    tensorrt_requires = [
        'tensorrt>=10.0.0',
        'cuda-python>=12.0.0'
    ]
else:
    tensorrt_requires = []

class CustomBuildCommand(build_py):
    """Custom build command to handle onnx-tensorrt."""
    def run(self) -> None:
        # Check if tensorrt extra is requested
        if any('tensorrt' in opt for opt in self.distribution.extras_require.get('tensorrt', [])):
            onnx_tensorrt_path = Path(__file__).parent / "third_party/onnx-tensorrt"
            
            if not onnx_tensorrt_path.exists():
                # Clone the repository if it doesn't exist
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/onnx/onnx-tensorrt.git",
                    str(onnx_tensorrt_path)
                ], check=True)

            # Build onnx-tensorrt
            build_dir = onnx_tensorrt_path / "build"
            build_dir.mkdir(exist_ok=True)
            
            subprocess.run([
                "cmake", "..",
                "-DCMAKE_BUILD_TYPE=Release"
            ], cwd=build_dir, check=True)

            subprocess.run([
                "cmake", "--build", ".", 
                "--config", "Release",
                "-j"
            ], cwd=build_dir, check=True)

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
