"""Defines the kinfer API."""

import os

if "ORT_DYLIB_PATH" not in os.environ:
    from pathlib import Path

    import onnxruntime as ort

    LIB_PATH = next((Path(ort.__file__).parent / "capi").glob("libonnxruntime.*"), None)
    if LIB_PATH is not None:
        os.environ["ORT_DYLIB_PATH"] = LIB_PATH.resolve().as_posix()

from .rust_bindings import get_version

__version__ = get_version()
