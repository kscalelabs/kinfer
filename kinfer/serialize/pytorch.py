"""Defines a serializer for PyTorch tensors."""

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    AudioFrameValue,
    CameraFrameSchema,
    CameraFrameValue,
    DType,
    IMUSchema,
    IMUValue,
    InputSchema,
    JointCommandsSchema,
    JointCommandsValue,
    JointCommandValue,
    JointPositionsSchema,
    JointPositionsValue,
    JointPositionValue,
    JointTorquesSchema,
    JointTorquesValue,
    JointTorqueValue,
    JointVelocitiesSchema,
    JointVelocitiesValue,
    JointVelocityValue,
    StateTensorSchema,
    StateTensorValue,
    TimestampSchema,
    TimestampValue,
    ValueSchema,
    VectorCommandSchema,
    VectorCommandValue,
)
from kinfer.serialize.base import (
    AudioFrameSerializer,
    CameraFrameSerializer,
    IMUSerializer,
    JointCommandsSerializer,
    JointPositionsSerializer,
    JointTorquesSerializer,
    JointVelocitiesSerializer,
    MultiSerializer,
    Serializer,
    StateTensorSerializer,
    TimestampSerializer,
    VectorCommandSerializer,
)
from kinfer.serialize.utils import (
    convert_angular_position,
    convert_angular_velocity,
    convert_torque,
    dtype_num_bytes,
    dtype_range,
    numpy_dtype,
    parse_bytes,
    pytorch_dtype,
)


def check_names_match(names_a: Sequence[str], names_b: Sequence[str]) -> None:
    name_set_a = set(names_a)
    name_set_b = set(names_b)
    if name_set_a != name_set_b:
        only_in_a = name_set_a - name_set_b
        only_in_b = name_set_b - name_set_a
        raise ValueError(f"Names must match: {only_in_a} != {only_in_b}")


class PyTorchBaseSerializer:
    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype


class PyTorchJointPositionsSerializer(PyTorchBaseSerializer, JointPositionsSerializer[Tensor]):
    def serialize_joint_positions(
        self,
        schema: JointPositionsSchema,
        value: JointPositionsValue,
    ) -> Tensor:
        names, values = schema.joint_names, value.values
        if set(names) != set(v.joint_name for v in values):
            raise ValueError(f"Number of joint names and values must match: {len(names)} != {len(values)}")
        value_map = {v.joint_name: v for v in values}
        tensor = torch.tensor(
            [convert_angular_position(value_map[name].value, value_map[name].unit, schema.unit) for name in names],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_positions(
        self,
        schema: JointPositionsSchema,
        value: Tensor,
    ) -> JointPositionsValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of tensor must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = value.detach().cpu().flatten().numpy().tolist()
        return JointPositionsValue(
            values=[
                JointPositionValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointVelocitiesSerializer(PyTorchBaseSerializer, JointVelocitiesSerializer[Tensor]):
    def serialize_joint_velocities(
        self,
        schema: JointVelocitiesSchema,
        value: JointVelocitiesValue,
    ) -> Tensor:
        names, values = schema.joint_names, value.values
        if set(names) != set(v.joint_name for v in values):
            raise ValueError(f"Number of joint names and values must match: {len(names)} != {len(values)}")
        value_map = {v.joint_name: v for v in values}
        tensor = torch.tensor(
            [convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit) for name in names],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_velocities(
        self,
        schema: JointVelocitiesSchema,
        value: Tensor,
    ) -> JointVelocitiesValue:
        value_list = value.detach().cpu().flatten().numpy().tolist()
        return JointVelocitiesValue(
            values=[
                JointVelocityValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointTorquesSerializer(PyTorchBaseSerializer, JointTorquesSerializer[Tensor]):
    def serialize_joint_torques(
        self,
        schema: JointTorquesSchema,
        value: JointTorquesValue,
    ) -> Tensor:
        names, values = schema.joint_names, value.values
        if set(names) != set(v.joint_name for v in values):
            raise ValueError(f"Number of joint names and values must match: {len(names)} != {len(values)}")
        value_map = {v.joint_name: v for v in values}
        tensor = torch.tensor(
            [convert_torque(value_map[name].value, value_map[name].unit, schema.unit) for name in names],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_torques(
        self,
        schema: JointTorquesSchema,
        value: Tensor,
    ) -> JointTorquesValue:
        value_list = value.detach().cpu().flatten().numpy().tolist()
        return JointTorquesValue(
            values=[
                JointTorqueValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointCommandsSerializer(PyTorchBaseSerializer, JointCommandsSerializer[Tensor]):
    def serialize_joint_commands(
        self,
        schema: JointCommandsSchema,
        value: JointCommandsValue,
    ) -> Tensor:
        names, values = schema.joint_names, value.values
        if set(names) != set(v.joint_name for v in values):
            raise ValueError(f"Number of joint names and values must match: {len(names)} != {len(values)}")
        value_map = {v.joint_name: v for v in values}
        tensor = torch.tensor(
            [
                convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit)
                for name in names
            ],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_commands(self, schema: JointCommandsSchema, value: Tensor) -> JointCommandsValue:
        value_list = value.detach().cpu().flatten().numpy().tolist()
        return JointCommandsValue(
            values=[
                JointCommandValue(
                    joint_name=name,
                )
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchCameraFrameSerializer(PyTorchBaseSerializer, CameraFrameSerializer[Tensor]):
    def serialize_camera_frame(self, schema: CameraFrameSchema, value: CameraFrameValue) -> Tensor:
        np_arr = parse_bytes(value.data, DType.UINT8)
        tensor = torch.from_numpy(np_arr).to(self.device, self.dtype) / 255.0
        if tensor.numel() != schema.channels * schema.height * schema.width:
            raise ValueError(
                "Length of data must match number of channels, height, and width: "
                f"{tensor.numel()} != {schema.channels} * {schema.height} * {schema.width}"
            )
        tensor = tensor.view(schema.channels, schema.height, schema.width)
        return tensor

    def deserialize_camera_frame(self, schema: CameraFrameSchema, value: Tensor) -> CameraFrameValue:
        np_arr = (value * 255.0).detach().cpu().flatten().numpy().astype(np.uint8)
        return CameraFrameValue(data=np_arr.tobytes())


class PyTorchAudioFrameSerializer(PyTorchBaseSerializer, AudioFrameSerializer[Tensor]):
    def serialize_audio_frame(self, schema: AudioFrameSchema, value: AudioFrameValue) -> Tensor:
        value_bytes = value.data
        if len(value_bytes) != schema.channels * schema.sample_rate * dtype_num_bytes(schema.dtype):
            raise ValueError(
                "Length of data must match number of channels, sample rate, and dtype: "
                f"{len(value_bytes)} != {schema.channels} * {schema.sample_rate} * {dtype_num_bytes(schema.dtype)}"
            )
        _, max_value = dtype_range(schema.dtype)
        np_arr = parse_bytes(value_bytes, schema.dtype)
        tensor = torch.from_numpy(np_arr).to(self.device, self.dtype)
        tensor = tensor.view(schema.channels, -1)
        tensor = tensor / max_value
        return tensor

    def deserialize_audio_frame(self, schema: AudioFrameSchema, value: Tensor) -> AudioFrameValue:
        _, max_value = dtype_range(schema.dtype)
        np_arr = (value * max_value).detach().cpu().flatten().numpy().astype(numpy_dtype(schema.dtype))
        return AudioFrameValue(data=np_arr.tobytes())


class PyTorchIMUSerializer(PyTorchBaseSerializer, IMUSerializer[Tensor]):
    def serialize_imu(self, schema: IMUSchema, value: IMUValue) -> Tensor:
        vectors: list[Tensor] = []
        if schema.use_accelerometer:
            vectors.append(
                torch.tensor(
                    [value.linear_acceleration.x, value.linear_acceleration.y, value.linear_acceleration.z],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        if schema.use_gyroscope:
            vectors.append(
                torch.tensor(
                    [value.angular_velocity.x, value.angular_velocity.y, value.angular_velocity.z],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        if schema.use_magnetometer:
            vectors.append(
                torch.tensor(
                    [value.magnetic_field.x, value.magnetic_field.y, value.magnetic_field.z],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        if not vectors:
            raise ValueError("IMU has nothing to serialize")
        return torch.stack(vectors, dim=0)

    def deserialize_imu(self, schema: IMUSchema, value: Tensor) -> IMUValue:
        vectors = value.tolist()
        imu_value = IMUValue()
        if schema.use_accelerometer:
            (x, y, z), vectors = vectors[0], vectors[1:]
            imu_value.linear_acceleration.x = x
            imu_value.linear_acceleration.y = y
            imu_value.linear_acceleration.z = z
        if schema.use_gyroscope:
            (x, y, z), vectors = vectors[0], vectors[1:]
            imu_value.angular_velocity.x = x
            imu_value.angular_velocity.y = y
            imu_value.angular_velocity.z = z
        if schema.use_magnetometer:
            (x, y, z), vectors = vectors[0], vectors[1:]
            imu_value.magnetic_field.x = x
            imu_value.magnetic_field.y = y
            imu_value.magnetic_field.z = z
        return imu_value


class PyTorchTimestampSerializer(PyTorchBaseSerializer, TimestampSerializer[Tensor]):
    def serialize_timestamp(self, schema: TimestampSchema, value: TimestampValue) -> Tensor:
        elapsed_seconds = value.seconds - schema.start_seconds
        elapsed_nanos = value.nanos - schema.start_nanos
        if elapsed_nanos < 0:
            elapsed_seconds -= 1
            elapsed_nanos += 1_000_000_000
        total_elapsed_seconds = elapsed_seconds + elapsed_nanos / 1_000_000_000
        return torch.tensor([total_elapsed_seconds], dtype=self.dtype, device=self.device, requires_grad=False)

    def deserialize_timestamp(self, schema: TimestampSchema, value: Tensor) -> TimestampValue:
        total_elapsed_seconds = value.item()
        elapsed_seconds = int(total_elapsed_seconds)
        elapsed_nanos = int((total_elapsed_seconds - elapsed_seconds) * 1_000_000_000)
        return TimestampValue(seconds=elapsed_seconds, nanos=elapsed_nanos)


class PyTorchVectorCommandSerializer(PyTorchBaseSerializer, VectorCommandSerializer[Tensor]):
    def serialize_vector_command(self, schema: VectorCommandSchema, value: VectorCommandValue) -> Tensor:
        return torch.tensor(value.values, dtype=self.dtype, device=self.device)

    def deserialize_vector_command(self, schema: VectorCommandSchema, value: Tensor) -> VectorCommandValue:
        return VectorCommandValue(values=value.tolist())


class PyTorchStateTensorSerializer(PyTorchBaseSerializer, StateTensorSerializer[Tensor]):
    def serialize_state_tensor(self, schema: StateTensorSchema, value: StateTensorValue) -> Tensor:
        value_bytes = value.data
        if len(value_bytes) != np.prod(schema.shape) * dtype_num_bytes(schema.dtype):
            raise ValueError(
                "Length of data must match number of elements: "
                f"{len(value_bytes)} != {np.prod(schema.shape)} * {dtype_num_bytes(schema.dtype)}"
            )
        np_arr = parse_bytes(value_bytes, schema.dtype)
        tensor = torch.from_numpy(np_arr).to(self.device, pytorch_dtype(schema.dtype))
        tensor = tensor.view(tuple(schema.shape))
        return tensor

    def deserialize_state_tensor(self, schema: StateTensorSchema, value: Tensor) -> StateTensorValue:
        return StateTensorValue(data=value.cpu().flatten().numpy().tobytes())


class PyTorchSerializer(
    PyTorchJointPositionsSerializer,
    PyTorchJointVelocitiesSerializer,
    PyTorchJointTorquesSerializer,
    PyTorchCameraFrameSerializer,
    PyTorchAudioFrameSerializer,
    PyTorchIMUSerializer,
    PyTorchTimestampSerializer,
    PyTorchVectorCommandSerializer,
    PyTorchStateTensorSerializer,
    Serializer[Tensor],
):
    def __init__(
        self,
        schema: ValueSchema,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        PyTorchBaseSerializer.__init__(self, device=device, dtype=dtype)
        Serializer.__init__(self, schema=schema)


class PyTorchInputSerializer(MultiSerializer[Tensor]):
    def __init__(self, schema: InputSchema) -> None:
        super().__init__([PyTorchSerializer(schema=s) for s in schema.inputs])
