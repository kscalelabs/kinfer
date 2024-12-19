"""Defines an interface for instantiating serializers."""

from typing import Literal

from kinfer import proto as P

from .base import MultiSerializer, Serializer
from .json import JsonMultiSerializer, JsonSerializer
from .numpy import NumpyMultiSerializer, NumpySerializer
from .pytorch import PyTorchMultiSerializer, PyTorchSerializer

SerializerType = Literal["json", "numpy", "pytorch"]


def get_serializer(schema: P.ValueSchema, serializer_type: SerializerType) -> Serializer:
    match serializer_type:
        case "json":
            return JsonSerializer(schema=schema)
        case "numpy":
            return NumpySerializer(schema=schema)
        case "pytorch":
            return PyTorchSerializer(schema=schema)
        case _:
            raise ValueError(f"Unsupported serializer type: {serializer_type}")


def get_multi_serializer(schema: P.IOSchema, serializer_type: SerializerType) -> MultiSerializer:
    match serializer_type:
        case "json":
            return JsonMultiSerializer(schema=schema)
        case "numpy":
            return NumpyMultiSerializer(schema=schema)
        case "pytorch":
            return PyTorchMultiSerializer(schema=schema)
        case _:
            raise ValueError(f"Unsupported serializer type: {serializer_type}")
