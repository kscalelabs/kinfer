"""Utility functions for serializing and deserializing Kinfer values."""

import numpy as np

from kinfer.protos.kinfer_pb2 import DType


def parse_bytes(data: bytes, dtype: DType) -> np.ndarray:
    match dtype:
        case DType.FP8:
            raise NotImplementedError("FP8 is not supported")
        case DType.FP16:
            return np.frombuffer(data, dtype=np.float16)
        case DType.FP32:
            return np.frombuffer(data, dtype=np.float32)
        case DType.FP64:
            return np.frombuffer(data, dtype=np.float64)
        case DType.INT8:
            return np.frombuffer(data, dtype=np.int8)
        case DType.INT16:
            return np.frombuffer(data, dtype=np.int16)
        case DType.INT32:
            return np.frombuffer(data, dtype=np.int32)
        case DType.INT64:
            return np.frombuffer(data, dtype=np.int64)
        case DType.UINT8:
            return np.frombuffer(data, dtype=np.uint8)
        case DType.UINT16:
            return np.frombuffer(data, dtype=np.uint16)
        case DType.UINT32:
            return np.frombuffer(data, dtype=np.uint32)
        case DType.UINT64:
            return np.frombuffer(data, dtype=np.uint64)
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


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
