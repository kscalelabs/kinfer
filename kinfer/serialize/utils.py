"""Utility functions for serializing and deserializing Kinfer values."""

import math
from typing import Any, Collection

import numpy as np
import torch

from kinfer import proto as P


def numpy_dtype(dtype: P.DType.ValueType) -> type[np.floating] | type[np.integer]:
    match dtype:
        case P.DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case P.DType.FP16:
            return np.float16
        case P.DType.FP32:
            return np.float32
        case P.DType.FP64:
            return np.float64
        case P.DType.INT8:
            return np.int8
        case P.DType.INT16:
            return np.int16
        case P.DType.INT32:
            return np.int32
        case P.DType.INT64:
            return np.int64
        case P.DType.UINT8:
            return np.uint8
        case P.DType.UINT16:
            return np.uint16
        case P.DType.UINT32:
            return np.uint32
        case P.DType.UINT64:
            return np.uint64
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def pytorch_dtype(dtype: P.DType.ValueType) -> torch.dtype:
    match dtype:
        case P.DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case P.DType.FP16:
            return torch.float16
        case P.DType.FP32:
            return torch.float32
        case P.DType.FP64:
            return torch.float64
        case P.DType.INT8:
            return torch.int8
        case P.DType.INT16:
            return torch.int16
        case P.DType.INT32:
            return torch.int32
        case P.DType.INT64:
            return torch.int64
        case P.DType.UINT8:
            return torch.uint8
        case P.DType.UINT16:
            return torch.uint16
        case P.DType.UINT32:
            return torch.uint32
        case P.DType.UINT64:
            return torch.uint64
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def parse_bytes(data: bytes, dtype: P.DType.ValueType) -> np.ndarray:
    return np.frombuffer(data, dtype=numpy_dtype(dtype))


def dtype_num_bytes(dtype: P.DType.ValueType) -> int:
    match dtype:
        case P.DType.FP8 | P.DType.INT8 | P.DType.UINT8:
            return 1
        case P.DType.FP16 | P.DType.INT16 | P.DType.UINT16:
            return 2
        case P.DType.FP32 | P.DType.INT32 | P.DType.UINT32:
            return 4
        case P.DType.FP64 | P.DType.INT64 | P.DType.UINT64:
            return 8
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_range(dtype: P.DType.ValueType) -> tuple[int, int]:
    match dtype:
        case P.DType.FP8:
            return -1, 1
        case P.DType.FP16:
            return -1, 1
        case P.DType.FP32:
            return -1, 1
        case P.DType.FP64:
            return -1, 1
        case P.DType.INT8:
            return -(2**7), 2**7 - 1
        case P.DType.INT16:
            return -(2**15), 2**15 - 1
        case P.DType.INT32:
            return -(2**31), 2**31 - 1
        case P.DType.INT64:
            return -(2**63), 2**63 - 1
        case P.DType.UINT8:
            return 0, 2**8 - 1
        case P.DType.UINT16:
            return 0, 2**16 - 1
        case P.DType.UINT32:
            return 0, 2**32 - 1
        case P.DType.UINT64:
            return 0, 2**64 - 1
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def convert_torque(
    value: float,
    from_unit: P.JointTorqueUnit.ValueType,
    to_unit: P.JointTorqueUnit.ValueType,
) -> float:
    if from_unit == to_unit:
        return value
    raise ValueError(f"Unsupported unit: {from_unit}")


def convert_angular_velocity(
    value: float,
    from_unit: P.JointVelocityUnit.ValueType,
    to_unit: P.JointVelocityUnit.ValueType,
) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == P.JointVelocityUnit.DEGREES_PER_SECOND:
        assert to_unit == P.JointVelocityUnit.RADIANS_PER_SECOND
        return value * math.pi / 180
    if from_unit == P.JointVelocityUnit.RADIANS_PER_SECOND:
        assert to_unit == P.JointVelocityUnit.DEGREES_PER_SECOND
        return value * 180 / math.pi
    raise ValueError(f"Unsupported unit: {from_unit}")


def convert_angular_position(
    value: float,
    from_unit: P.JointPositionUnit.ValueType,
    to_unit: P.JointPositionUnit.ValueType,
) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == P.JointPositionUnit.DEGREES:
        return value * math.pi / 180
    if from_unit == P.JointPositionUnit.RADIANS:
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
