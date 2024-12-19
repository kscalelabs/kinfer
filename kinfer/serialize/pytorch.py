"""Defines a serializer for PyTorch tensors."""

from typing import cast

import numpy as np
import torch
from torch import Tensor

from kinfer import proto as P
from kinfer.serialize.base import (
    AudioFrameSerializer,
    CameraFrameSerializer,
    ImuSerializer,
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
    check_names_match,
    convert_angular_position,
    convert_angular_velocity,
    convert_torque,
    dtype_num_bytes,
    dtype_range,
    numpy_dtype,
    parse_bytes,
    pytorch_dtype,
)


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
        schema: P.JointPositionsSchema,
        value: P.JointPositionsValue,
    ) -> Tensor:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        tensor = torch.tensor(
            [
                convert_angular_position(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_positions(
        self,
        schema: P.JointPositionsSchema,
        value: Tensor,
    ) -> P.JointPositionsValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of tensor must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = cast(list[float], value.detach().cpu().numpy().astype(float).tolist())
        return P.JointPositionsValue(
            values=[
                P.JointPositionValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointVelocitiesSerializer(PyTorchBaseSerializer, JointVelocitiesSerializer[Tensor]):
    def serialize_joint_velocities(
        self,
        schema: P.JointVelocitiesSchema,
        value: P.JointVelocitiesValue,
    ) -> Tensor:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        tensor = torch.tensor(
            [
                convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_velocities(
        self,
        schema: P.JointVelocitiesSchema,
        value: Tensor,
    ) -> P.JointVelocitiesValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of tensor must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = cast(list[float], value.detach().cpu().numpy().astype(float).tolist())
        return P.JointVelocitiesValue(
            values=[
                P.JointVelocityValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointTorquesSerializer(PyTorchBaseSerializer, JointTorquesSerializer[Tensor]):
    def serialize_joint_torques(
        self,
        schema: P.JointTorquesSchema,
        value: P.JointTorquesValue,
    ) -> Tensor:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        tensor = torch.tensor(
            [convert_torque(value_map[name].value, value_map[name].unit, schema.unit) for name in schema.joint_names],
            dtype=self.dtype,
            device=self.device,
        )
        return tensor

    def deserialize_joint_torques(
        self,
        schema: P.JointTorquesSchema,
        value: Tensor,
    ) -> P.JointTorquesValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of tensor must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = cast(list[float], value.detach().cpu().numpy().astype(float).tolist())
        return P.JointTorquesValue(
            values=[
                P.JointTorqueValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointCommandsSerializer(PyTorchBaseSerializer, JointCommandsSerializer[Tensor]):
    def _convert_value_to_tensor(
        self,
        value: P.JointCommandValue,
        schema: P.JointCommandsSchema,
    ) -> Tensor:
        return torch.tensor(
            [
                convert_torque(value.torque, value.torque_unit, schema.torque_unit),
                convert_angular_velocity(value.velocity, value.velocity_unit, schema.velocity_unit),
                convert_angular_position(value.position, value.position_unit, schema.position_unit),
                value.kp,
                value.kd,
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def _convert_tensor_to_value(
        self,
        values: list[float],
        schema: P.JointCommandsSchema,
        name: str,
    ) -> P.JointCommandValue:
        if len(values) != 5:
            raise ValueError(f"Shape of tensor must match number of joint commands: {len(values)} != 5")
        return P.JointCommandValue(
            joint_name=name,
            torque=values[0],
            velocity=values[1],
            position=values[2],
            kp=values[3],
            kd=values[4],
            torque_unit=schema.torque_unit,
            velocity_unit=schema.velocity_unit,
            position_unit=schema.position_unit,
        )

    def serialize_joint_commands(
        self,
        schema: P.JointCommandsSchema,
        value: P.JointCommandsValue,
    ) -> Tensor:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        tensor = torch.stack(
            [self._convert_value_to_tensor(value_map[name], schema) for name in schema.joint_names],
            dim=0,
        )
        return tensor

    def deserialize_joint_commands(self, schema: P.JointCommandsSchema, value: Tensor) -> P.JointCommandsValue:
        if value.shape != (len(schema.joint_names), 5):
            raise ValueError(
                "Shape of tensor must match number of joint names and commands: "
                f"{value.shape} != ({len(schema.joint_names)}, 5)"
            )
        value_list = cast(list[list[float]], value.detach().cpu().numpy().astype(float).tolist())
        return P.JointCommandsValue(
            values=[
                self._convert_tensor_to_value(value_list[i], schema, name) for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchCameraFrameSerializer(PyTorchBaseSerializer, CameraFrameSerializer[Tensor]):
    def serialize_camera_frame(self, schema: P.CameraFrameSchema, value: P.CameraFrameValue) -> Tensor:
        np_arr = parse_bytes(value.data, P.DType.UINT8)
        tensor = torch.from_numpy(np_arr).to(self.device, self.dtype) / 255.0
        if tensor.numel() != schema.channels * schema.height * schema.width:
            raise ValueError(
                "Length of data must match number of channels, height, and width: "
                f"{tensor.numel()} != {schema.channels} * {schema.height} * {schema.width}"
            )
        tensor = tensor.view(schema.channels, schema.height, schema.width)
        return tensor

    def deserialize_camera_frame(self, schema: P.CameraFrameSchema, value: Tensor) -> P.CameraFrameValue:
        np_arr = (value * 255.0).detach().cpu().flatten().numpy().astype(np.uint8)
        return P.CameraFrameValue(data=np_arr.tobytes())


class PyTorchAudioFrameSerializer(PyTorchBaseSerializer, AudioFrameSerializer[Tensor]):
    def serialize_audio_frame(self, schema: P.AudioFrameSchema, value: P.AudioFrameValue) -> Tensor:
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

    def deserialize_audio_frame(self, schema: P.AudioFrameSchema, value: Tensor) -> P.AudioFrameValue:
        _, max_value = dtype_range(schema.dtype)
        np_arr = (value * max_value).detach().cpu().flatten().numpy().astype(numpy_dtype(schema.dtype))
        return P.AudioFrameValue(data=np_arr.tobytes())


class PyTorchImuSerializer(PyTorchBaseSerializer, ImuSerializer[Tensor]):
    def serialize_imu(self, schema: P.ImuSchema, value: P.ImuValue) -> Tensor:
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

    def deserialize_imu(self, schema: P.ImuSchema, value: Tensor) -> P.ImuValue:
        vectors = value.tolist()
        imu_value = P.ImuValue()
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
    def serialize_timestamp(self, schema: P.TimestampSchema, value: P.TimestampValue) -> Tensor:
        elapsed_seconds = value.seconds - schema.start_seconds
        elapsed_nanos = value.nanos - schema.start_nanos
        if elapsed_nanos < 0:
            elapsed_seconds -= 1
            elapsed_nanos += 1_000_000_000
        total_elapsed_seconds = elapsed_seconds + elapsed_nanos / 1_000_000_000
        return torch.tensor([total_elapsed_seconds], dtype=self.dtype, device=self.device, requires_grad=False)

    def deserialize_timestamp(self, schema: P.TimestampSchema, value: Tensor) -> P.TimestampValue:
        total_elapsed_seconds = value.item()
        elapsed_seconds = int(total_elapsed_seconds)
        elapsed_nanos = int((total_elapsed_seconds - elapsed_seconds) * 1_000_000_000)
        return P.TimestampValue(seconds=elapsed_seconds, nanos=elapsed_nanos)


class PyTorchVectorCommandSerializer(PyTorchBaseSerializer, VectorCommandSerializer[Tensor]):
    def serialize_vector_command(self, schema: P.VectorCommandSchema, value: P.VectorCommandValue) -> Tensor:
        return torch.tensor(value.values, dtype=self.dtype, device=self.device)

    def deserialize_vector_command(self, schema: P.VectorCommandSchema, value: Tensor) -> P.VectorCommandValue:
        if value.shape != (schema.dimensions,):
            raise ValueError(f"Shape of tensor must match number of dimensions: {value.shape} != {schema.dimensions}")
        values = cast(list[float], value.tolist())
        return P.VectorCommandValue(values=values)


class PyTorchStateTensorSerializer(PyTorchBaseSerializer, StateTensorSerializer[Tensor]):
    def serialize_state_tensor(self, schema: P.StateTensorSchema, value: P.StateTensorValue) -> Tensor:
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

    def deserialize_state_tensor(self, schema: P.StateTensorSchema, value: Tensor) -> P.StateTensorValue:
        return P.StateTensorValue(data=value.detach().cpu().flatten().numpy().tobytes())


class PyTorchSerializer(
    PyTorchJointPositionsSerializer,
    PyTorchJointVelocitiesSerializer,
    PyTorchJointTorquesSerializer,
    PyTorchJointCommandsSerializer,
    PyTorchCameraFrameSerializer,
    PyTorchAudioFrameSerializer,
    PyTorchImuSerializer,
    PyTorchTimestampSerializer,
    PyTorchVectorCommandSerializer,
    PyTorchStateTensorSerializer,
    Serializer[Tensor],
):
    def __init__(
        self,
        schema: P.ValueSchema,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        PyTorchBaseSerializer.__init__(self, device=device, dtype=dtype)
        Serializer.__init__(self, schema=schema)


class PyTorchMultiSerializer(MultiSerializer[Tensor]):
    def __init__(self, schema: P.IOSchema) -> None:
        super().__init__([PyTorchSerializer(schema=s) for s in schema.values])
