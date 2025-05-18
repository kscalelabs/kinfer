# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re
import subprocess
from typing import List

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("kinfer/requirements.txt", "r", encoding="utf-8") as f:
    requirements: List[str] = f.read().splitlines()


with open("Cargo.toml", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^version = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in Cargo.toml"
version: str = version_re.group(1)

requirements_dev = [
    "black",
    "darglint",
    "mypy",
    "pytest",
    "ruff",
    "types-tensorflow",
]

requirements_pytorch = [
    "torch",
]

requirements_jax = [
    "tensorflow",
    "tf2onnx>=1.16.0",
    "jax",
    "equinox",
    "numpy<2",
]


class RustBuildExt(build_ext):
    def run(self) -> None:
        subprocess.run(["cargo", "run", "--bin", "stub_gen"], check=True)
        super().run()


class CustomBuild(build_py):
    def run(self) -> None:
        self.run_command("build_ext")
        super().run()


setup(
    name="kinfer",
    version=version,
    description="Tool to make it easier to run a model on a real robot",
    author="K-Scale Labs",
    url="https://github.com/kscalelabs/kinfer.git",
    rust_extensions=[
        RustExtension(
            target="kinfer.rust_bindings",
            path="kinfer/rust_bindings/Cargo.toml",
            binding=Binding.PyO3,
        ),
    ],
    setup_requires=["setuptools-rust"],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
        "pytorch": requirements_pytorch,
        "jax": requirements_jax,
        "all": requirements_dev + requirements_pytorch + requirements_jax,
    },
    include_package_data=True,
    packages=find_packages(),
    cmdclass={
        "build_ext": RustBuildExt,
        "build_py": CustomBuild,
    },
)
