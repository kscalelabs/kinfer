# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re
from typing import List

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("kinfer/requirements.txt", "r", encoding="utf-8") as f:
    requirements: List[str] = f.read().splitlines()


with open("kinfer/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: List[str] = f.read().splitlines()


with open("kinfer/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in kinfer/__init__.py"
version: str = version_re.group(1)


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
    packages=find_packages(),
    # entry_points={
    #     "console_scripts": [
    #         "kinfer.cli:main",
    #     ],
    # },
)
