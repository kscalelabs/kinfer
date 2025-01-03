"""Defines the kinfer API."""

from . import proto as K
from .export.pytorch import export_model, get_model
from .inference.base import KModel
from .inference.python import ONNXModel
from .rust_bindings import get_version
from .serialize import get_multi_serializer, get_serializer

__version__ = get_version()
