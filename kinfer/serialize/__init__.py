"""Defines an interface for instantiating serializers."""

from typing import Literal, overload

from kinfer import proto as K

from .base import MultiSerializer, Serializer
from .json import JsonMultiSerializer, JsonSerializer
from .numpy import NumpyMultiSerializer, NumpySerializer
from .pytorch import PyTorchMultiSerializer, PyTorchSerializer

SerializerType = Literal["json", "numpy", "pytorch"]


@overload
def get_serializer(schema: K.ValueSchema, serializer_type: Literal["json"]) -> JsonSerializer: ...


@overload
def get_serializer(schema: K.ValueSchema, serializer_type: Literal["numpy"]) -> NumpySerializer: ...


@overload
def get_serializer(schema: K.ValueSchema, serializer_type: Literal["pytorch"]) -> PyTorchSerializer: ...


def get_serializer(schema: K.ValueSchema, serializer_type: SerializerType) -> Serializer:
    match serializer_type:
        case "json":
            return JsonSerializer(schema=schema)
        case "numpy":
            return NumpySerializer(schema=schema)
        case "pytorch":
            return PyTorchSerializer(schema=schema)
        case _:
            raise ValueError(f"Unsupported serializer type: {serializer_type}")


@overload
def get_multi_serializer(schema: K.IOSchema, serializer_type: Literal["json"]) -> JsonMultiSerializer: ...


@overload
def get_multi_serializer(schema: K.IOSchema, serializer_type: Literal["numpy"]) -> NumpyMultiSerializer: ...


@overload
def get_multi_serializer(schema: K.IOSchema, serializer_type: Literal["pytorch"]) -> PyTorchMultiSerializer: ...


def get_multi_serializer(schema: K.IOSchema, serializer_type: SerializerType) -> MultiSerializer:
    match serializer_type:
        case "json":
            return JsonMultiSerializer(schema=schema)
        case "numpy":
            return NumpyMultiSerializer(schema=schema)
        case "pytorch":
            return PyTorchMultiSerializer(schema=schema)
        case _:
            raise ValueError(f"Unsupported serializer type: {serializer_type}")
