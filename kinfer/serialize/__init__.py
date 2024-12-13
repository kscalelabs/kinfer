"""Defines an interface for instantiating serializers."""

from typing import Literal

from kinfer.protos.kinfer_pb2 import InputSchema, OutputSchema

from .base import MultiSerializer, Serializer
from .json import JsonMultiSerializer, JsonSerializer
from .numpy import NumpyMultiSerializer, NumpySerializer
from .pytorch import PyTorchMultiSerializer, PyTorchSerializer

SerializerType = Literal["json", "numpy", "pytorch"]


def get_serializer(serializer_type: SerializerType) -> Serializer:
    match serializer_type:
        case "json":
            return JsonSerializer()
        case "numpy":
            return NumpySerializer()
        case "pytorch":
            return PyTorchSerializer()
        case _:
            raise ValueError(f"Unsupported serializer type: {serializer_type}")


def get_multi_serializer(schema: InputSchema | OutputSchema, serializer_type: SerializerType) -> MultiSerializer:
    match serializer_type:
        case "json":
            return JsonMultiSerializer(schema=schema)
        case "numpy":
            return NumpyMultiSerializer(schema=schema)
        case "pytorch":
            return PyTorchMultiSerializer(schema=schema)
        case _:
            raise ValueError(f"Unsupported serializer type: {serializer_type}")
