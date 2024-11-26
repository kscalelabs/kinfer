# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import os
import platform
import re
import sys
from subprocess import check_call
from typing import List

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

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
            print("INFO: Running platform check...")
            check_call([sys.executable, "post_install.py"])
        except Exception as e:
            print(f"WARNING: Platform check failed: {e}")

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
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev},
    packages=find_packages(exclude=exclude_packages),
    entry_points={
        "console_scripts": [
            "kinfer-check=kinfer.platform_check:check_platform_specific_modules",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
)
