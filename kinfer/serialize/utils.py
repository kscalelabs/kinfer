"""Utility functions for serializing and deserializing Kinfer values."""

import math
from typing import Sequence

import numpy as np
import torch

from kinfer.protos.kinfer_pb2 import DType, JointPositionUnit, JointTorqueUnit, JointVelocityUnit


def numpy_dtype(dtype: DType) -> type[np.floating] | type[np.integer]:
    match dtype:
        case DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case DType.FP16:
            return np.float16
        case DType.FP32:
            return np.float32
        case DType.FP64:
            return np.float64
        case DType.INT8:
            return np.int8
        case DType.INT16:
            return np.int16
        case DType.INT32:
            return np.int32
        case DType.INT64:
            return np.int64
        case DType.UINT8:
            return np.uint8
        case DType.UINT16:
            return np.uint16
        case DType.UINT32:
            return np.uint32
        case DType.UINT64:
            return np.uint64
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def pytorch_dtype(dtype: DType) -> torch.dtype:
    match dtype:
        case DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case DType.FP16:
            return torch.float16
        case DType.FP32:
            return torch.float32
        case DType.FP64:
            return torch.float64
        case DType.INT8:
            return torch.int8
        case DType.INT16:
            return torch.int16
        case DType.INT32:
            return torch.int32
        case DType.INT64:
            return torch.int64
        case DType.UINT8:
            return torch.uint8
        case DType.UINT16:
            return torch.uint16
        case DType.UINT32:
            return torch.uint32
        case DType.UINT64:
            return torch.uint64
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def parse_bytes(data: bytes, dtype: DType) -> np.ndarray:
    return np.frombuffer(data, dtype=numpy_dtype(dtype))


def dtype_num_bytes(dtype: DType) -> int:
    match dtype:
        case DType.FP8 | DType.INT8 | DType.UINT8:
            return 1
        case DType.FP16 | DType.INT16 | DType.UINT16:
            return 2
        case DType.FP32 | DType.INT32 | DType.UINT32:
            return 4
        case DType.FP64 | DType.INT64 | DType.UINT64:
            return 8
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def dtype_range(dtype: DType) -> tuple[int, int]:
    match dtype:
        case DType.FP8:
            return -1, 1
        case DType.FP16:
            return -1, 1
        case DType.FP32:
            return -1, 1
        case DType.FP64:
            return -1, 1
        case DType.INT8:
            return -(2**7), 2**7 - 1
        case DType.INT16:
            return -(2**15), 2**15 - 1
        case DType.INT32:
            return -(2**31), 2**31 - 1
        case DType.INT64:
            return -(2**63), 2**63 - 1
        case DType.UINT8:
            return 0, 2**8 - 1
        case DType.UINT16:
            return 0, 2**16 - 1
        case DType.UINT32:
            return 0, 2**32 - 1
        case DType.UINT64:
            return 0, 2**64 - 1
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def convert_torque(value: float, from_unit: JointTorqueUnit, to_unit: JointTorqueUnit) -> float:
    if from_unit == to_unit:
        return value
    raise ValueError(f"Unsupported unit: {from_unit}")


def convert_angular_velocity(value: float, from_unit: JointVelocityUnit, to_unit: JointVelocityUnit) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == JointVelocityUnit.DEGREES_PER_SECOND:
        assert to_unit == JointVelocityUnit.RADIANS_PER_SECOND
        return value * math.pi / 180
    if from_unit == JointVelocityUnit.RADIANS_PER_SECOND:
        assert to_unit == JointVelocityUnit.DEGREES_PER_SECOND
        return value * 180 / math.pi
    raise ValueError(f"Unsupported unit: {from_unit}")


def convert_angular_position(value: float, from_unit: JointPositionUnit, to_unit: JointPositionUnit) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == JointPositionUnit.DEGREES:
        return value * math.pi / 180
    if from_unit == JointPositionUnit.RADIANS:
        return value * 180 / math.pi
    raise ValueError(f"Unsupported unit: {from_unit}")


def check_names_match(a_name: str, a: Sequence[str], b_name: str, b: Sequence[str]) -> None:
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
