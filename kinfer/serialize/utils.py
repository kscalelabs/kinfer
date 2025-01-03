"""Utility functions for serializing and deserializing Kinfer values."""

import math
from typing import Any, Collection

import numpy as np
import torch

from kinfer import proto as K


def numpy_dtype(dtype: K.DType.ValueType) -> type[np.floating] | type[np.integer]:
    match dtype:
        case K.DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case K.DType.FP16:
            return np.float16
        case K.DType.FP32:
            return np.float32
        case K.DType.FP64:
            return np.float64
        case K.DType.INT8:
            return np.int8
        case K.DType.INT16:
            return np.int16
        case K.DType.INT32:
            return np.int32
        case K.DType.INT64:
            return np.int64
        case K.DType.UINT8:
            return np.uint8
        case K.DType.UINT16:
            return np.uint16
        case K.DType.UINT32:
            return np.uint32
        case K.DType.UINT64:
            return np.uint64
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def pytorch_dtype(dtype: K.DType.ValueType) -> torch.dtype:
    match dtype:
        case K.DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case K.DType.FP16:
            return torch.float16
        case K.DType.FP32:
            return torch.float32
        case K.DType.FP64:
            return torch.float64
        case K.DType.INT8:
            return torch.int8
        case K.DType.INT16:
            return torch.int16
        case K.DType.INT32:
            return torch.int32
        case K.DType.INT64:
            return torch.int64
        case K.DType.UINT8:
            return torch.uint8
        case K.DType.UINT16:
            return torch.uint16
        case K.DType.UINT32:
            return torch.uint32
        case K.DType.UINT64:
            return torch.uint64
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def parse_bytes(data: bytes, dtype: K.DType.ValueType) -> np.ndarray:
    return np.frombuffer(data, dtype=numpy_dtype(dtype))


def dtype_num_bytes(dtype: K.DType.ValueType) -> int:
    match dtype:
        case K.DType.FP8 | K.DType.INT8 | K.DType.UINT8:
            return 1
        case K.DType.FP16 | K.DType.INT16 | K.DType.UINT16:
            return 2
        case K.DType.FP32 | K.DType.INT32 | K.DType.UINT32:
            return 4
        case K.DType.FP64 | K.DType.INT64 | K.DType.UINT64:
            return 8
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_range(dtype: K.DType.ValueType) -> tuple[int, int]:
    match dtype:
        case K.DType.FP8:
            return -1, 1
        case K.DType.FP16:
            return -1, 1
        case K.DType.FP32:
            return -1, 1
        case K.DType.FP64:
            return -1, 1
        case K.DType.INT8:
            return -(2**7), 2**7 - 1
        case K.DType.INT16:
            return -(2**15), 2**15 - 1
        case K.DType.INT32:
            return -(2**31), 2**31 - 1
        case K.DType.INT64:
            return -(2**63), 2**63 - 1
        case K.DType.UINT8:
            return 0, 2**8 - 1
        case K.DType.UINT16:
            return 0, 2**16 - 1
        case K.DType.UINT32:
            return 0, 2**32 - 1
        case K.DType.UINT64:
            return 0, 2**64 - 1
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def convert_torque(
    value: float,
    from_unit: K.JointTorqueUnit.ValueType,
    to_unit: K.JointTorqueUnit.ValueType,
) -> float:
    if from_unit == to_unit:
        return value
    raise ValueError(f"Unsupported unit: {from_unit}")


def convert_angular_velocity(
    value: float,
    from_unit: K.JointVelocityUnit.ValueType,
    to_unit: K.JointVelocityUnit.ValueType,
) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == K.JointVelocityUnit.DEGREES_PER_SECOND:
        assert to_unit == K.JointVelocityUnit.RADIANS_PER_SECOND
        return value * math.pi / 180
    if from_unit == K.JointVelocityUnit.RADIANS_PER_SECOND:
        assert to_unit == K.JointVelocityUnit.DEGREES_PER_SECOND
        return value * 180 / math.pi
    raise ValueError(f"Unsupported unit: {from_unit}")


def convert_angular_position(
    value: float,
    from_unit: K.JointPositionUnit.ValueType,
    to_unit: K.JointPositionUnit.ValueType,
) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == K.JointPositionUnit.DEGREES:
        return value * math.pi / 180
    if from_unit == K.JointPositionUnit.RADIANS:
        return value * 180 / math.pi
    raise ValueError(f"Unsupported unit: {from_unit}")


def check_names_match(a_name: str, a: Collection[str], b_name: str, b: Collection[str]) -> None:
    name_set_a = set(a)
    name_set_b = set(b)
    if name_set_a != name_set_b:
        only_in_a = name_set_a - name_set_b
        only_in_b = name_set_b - name_set_a
        message = "Names must match!"
        if only_in_a:
            message += f" Only in {a_name}: {only_in_a}"
        if only_in_b:
            message += f" Only in {b_name}: {only_in_b}"
        raise ValueError(message)


def as_float(value: Any) -> float:  # noqa: ANN401
    if not isinstance(value, (float, int)):
        raise ValueError(f"Value must be a float or int: {value}")
    return float(value)
