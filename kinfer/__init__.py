"""Defines the kinfer API."""

from .rust_bindings import get_version

__version__ = get_version()
