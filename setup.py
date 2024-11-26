# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import os
import platform
import re
import sys
from subprocess import check_call
from typing import List

from setuptools import find_packages, setup, Extension
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
import subprocess
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()

PLATFORM = platform.system().lower()

# Define platform-specific exclusions
if PLATFORM == "darwin":
    exclude_packages = ["kinfer.optimize.tensorrt*"]
else:
    exclude_packages = []

def should_include_requirement(platform_spec: str) -> bool:
    """Check if requirement should be included based on platform specification."""
    # Extract the platform systems from the spec (e.g., "Linux", "Windows")
    match = re.search(r'platform_system\s*in\s*\(([^)]+)\)', platform_spec)
    if not match:
        return True

    # Parse the platforms
    platforms = [p.strip(' "\'') for p in match.group(1).split(',')]
    return PLATFORM in [p.lower() for p in platforms]

# Filter requirements based on platform
requirements = []
with open("kinfer/requirements.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Split requirement into package and platform specification
        parts = line.split(';')
        package = parts[0].strip()

        # Check platform specification if it exists
        if len(parts) > 1 and not should_include_requirement(parts[1].strip()):
            continue

        requirements.append(package)

with open("kinfer/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: List[str] = f.read().splitlines()

with open("kinfer/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in kinfer/__init__.py"
version: str = version_re.group(1)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self) -> None:
        install.run(self)
        try:
            # Build TensorRT parser if tensorrt extra is requested
            if any('tensorrt' in opt for opt in getattr(self.distribution, 'extras_require', {}).get('tensorrt', [])):
                from kinfer.optimize.tensorrt_build import build_tensorrt_parser
                build_tensorrt_parser()

            print("INFO: Running platform check...")
            check_call([sys.executable, "post_install.py"])
        except Exception as e:
            print(f"WARNING: Platform check failed: {e}")

def get_extensions() -> List[Extension]:
    """Get platform-specific extensions."""
    extensions = []

    # Only add TensorRT extension for Linux/Windows
    if PLATFORM in ("linux", "windows"):
        try:
            from kinfer.optimize.tensorrt_build import check_tensorrt_compatibility
            if check_tensorrt_compatibility():
                extensions.append(
                    Extension(
                        "kinfer.optimize.tensorrt._C",
                        sources=[],  # Will be built by custom command
                        optional=True  # Makes the build continue even if this fails
                    )
                )
        except ImportError:
            print("WARNING: TensorRT build utilities not available")
    
    return extensions

class CustomBuildCommand(build_py):
    """Custom build command to handle git submodules."""
    def run(self) -> None:
        if not (Path(__file__).parent / "third_party/onnx-tensorrt").exists():
            # Initialize submodules
            subprocess.run(["git", "submodule", "init"], check=True)
            subprocess.run(["git", "submodule", "update"], check=True)

        super().run()

class CustomSdistCommand(sdist):
    """Custom sdist command to handle git submodules."""
    def run(self) -> None:
        if not (Path(__file__).parent / "third_party/onnx-tensorrt").exists():
            # Initialize submodules
            subprocess.run(["git", "submodule", "init"], check=True)
            subprocess.run(["git", "submodule", "update"], check=True)

        super().run()

setup(
    name="kinfer",
    version=version,
    description="The kinfer project",
    author="K-Scale Labs",
    url="https://github.com/kscalelabs/kinfer.git",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'tensorrt': [
            'tensorrt>=10.6.0; platform_system=="Linux" or platform_system=="Windows"',
            'cuda-python>=12.0.0; platform_system=="Linux" or platform_system=="Windows"',
        ],
        'dev': requirements_dev,
    },
    packages=find_packages(exclude=exclude_packages),
    entry_points={
        "console_scripts": [
            "kinfer-check=kinfer.platform_check:check_platform_specific_modules",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
        "build_py": CustomBuildCommand,
        "sdist": CustomSdistCommand,
    },
    ext_modules=get_extensions(),
)
