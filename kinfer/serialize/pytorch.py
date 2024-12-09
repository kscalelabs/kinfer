"""Defines a serializer for PyTorch tensors."""

import math
from typing import Sequence

import torch
from torch import Tensor

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    AudioFrameValue,
    CameraFrameSchema,
    CameraFrameValue,
    IMUSchema,
    IMUValue,
    InputSchema,
    JointPositionsSchema,
    JointPositionsValue,
    JointPositionUnit,
    JointPositionValue,
    JointTorquesSchema,
    JointTorquesValue,
    JointTorqueUnit,
    JointTorqueValue,
    JointVelocitiesSchema,
    JointVelocitiesValue,
    JointVelocityUnit,
    JointVelocityValue,
    TensorSchema,
    TensorValue,
    TimestampSchema,
    TimestampValue,
    ValueSchema,
)
from kinfer.serialize.base import (
    AudioFrameSerializer,
    CameraFrameSerializer,
    IMUSerializer,
    JointPositionsSerializer,
    JointTorquesSerializer,
    JointVelocitiesSerializer,
    MultiSerializer,
    Serializer,
    TensorSerializer,
    TimestampSerializer,
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


class PyTorchTensorSerializer(PyTorchBaseSerializer, TensorSerializer[Tensor]):
    def serialize_tensor(self, schema: TensorSchema, value: TensorValue) -> Tensor:
        tensor = torch.tensor(value.data, dtype=self.dtype, device=self.device)
        return tensor.view(*schema.shape)

    def deserialize_tensor(self, schema: TensorSchema, value: Tensor) -> TensorValue:
        schema_shape, value_shape = tuple(schema.shape), tuple(value.shape)
        if schema_shape != value_shape:
            raise ValueError(f"Shape of tensor must match schema: {value_shape} != {schema_shape}")
        tensor_data = value.detach().cpu().flatten().numpy().tolist()
        return TensorValue(data=tensor_data)


class PyTorchJointPositionsSerializer(PyTorchBaseSerializer, JointPositionsSerializer[Tensor]):
    def _convert_angular_position(
        self,
        value: float,
        from_unit: JointPositionUnit,
        to_unit: JointPositionUnit,
    ) -> float:
        if from_unit == to_unit:
            return value
        if from_unit == JointPositionUnit.DEGREES:
            return value * math.pi / 180
        if from_unit == JointPositionUnit.RADIANS:
            return value * 180 / math.pi
        raise ValueError(f"Unsupported unit: {from_unit}")

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
            [
                self._convert_angular_position(value_map[name].value, value_map[name].unit, schema.unit)
                for name in names
            ],
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
                JointPositionValue(
                    joint_name=name,
                    value=value_list[i],
                    unit=schema.unit,
                )
                for i, name in enumerate(schema.joint_names)
            ]
        )


class PyTorchJointVelocitiesSerializer(PyTorchBaseSerializer, JointVelocitiesSerializer[Tensor]):
    def _convert_angular_velocity(
        self,
        value: float,
        from_unit: JointVelocityUnit,
        to_unit: JointVelocityUnit,
    ) -> float:
        if from_unit == to_unit:
            return value
        if from_unit == JointVelocityUnit.DEGREES_PER_SECOND:
            assert to_unit == JointVelocityUnit.RADIANS_PER_SECOND
            return value * math.pi / 180
        if from_unit == JointVelocityUnit.RADIANS_PER_SECOND:
            assert to_unit == JointVelocityUnit.DEGREES_PER_SECOND
            return value * 180 / math.pi
        raise ValueError(f"Unsupported unit: {from_unit}")

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
            [
                self._convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit)
                for name in names
            ],
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
    def _convert_torque(
        self,
        value: float,
        from_unit: JointTorqueUnit,
        to_unit: JointTorqueUnit,
    ) -> float:
        if from_unit == to_unit:
            return value
        raise ValueError(f"Unsupported unit: {from_unit}")

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
            [self._convert_torque(value_map[name].value, value_map[name].unit, schema.unit) for name in names],
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


class PyTorchCameraFrameSerializer(PyTorchBaseSerializer, CameraFrameSerializer[Tensor]):
    def serialize_camera_frame(self, schema: CameraFrameSchema, value: CameraFrameValue) -> Tensor:
        value_bytes = value.data
        assert len(value_bytes) == schema.channels * schema.height * schema.width
        value_list = [float(b) / 255.0 for b in value_bytes]
        tensor = torch.tensor(value_list, dtype=self.dtype, device=self.device)
        tensor = tensor.view(schema.channels, schema.height, schema.width)
        return tensor

    def deserialize_camera_frame(self, schema: CameraFrameSchema, value: Tensor) -> CameraFrameValue:
        camera_data = value.cpu().flatten().numpy().tolist()
        camera_bytes = bytes([int(round(a * 255)) for a in camera_data])
        return CameraFrameValue(data=camera_bytes)


class PyTorchAudioFrameSerializer(PyTorchBaseSerializer, AudioFrameSerializer[Tensor]):
    def serialize_audio_frame(self, schema: AudioFrameSchema, value: AudioFrameValue) -> Tensor:
        value_bytes = value.data
        assert len(value_bytes) == schema.channels * schema.sample_rate * schema.bytes_per_sample
        value_list = [float(b) / 255.0 for b in value_bytes]
        tensor = torch.tensor(value_list, dtype=self.dtype, device=self.device)
        tensor = tensor.view(schema.channels, -1)
        return tensor

    def deserialize_audio_frame(self, schema: AudioFrameSchema, value: Tensor) -> AudioFrameValue:
        audio_data = value.cpu().flatten().numpy().tolist()
        audio_bytes = bytes([int(round(a * 255)) for a in audio_data])
        return AudioFrameValue(data=audio_bytes)


class PyTorchIMUSerializer(PyTorchBaseSerializer, IMUSerializer[Tensor]):
    def serialize_imu(self, schema: IMUSchema, value: IMUValue) -> Tensor:
        return torch.tensor(value.data, dtype=self.dtype, device=self.device)

    def deserialize_imu(self, schema: IMUSchema, value: Tensor) -> IMUValue:
        return IMUValue(schema=schema, data=value.tolist())


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


class PyTorchSerializer(
    PyTorchTensorSerializer,
    PyTorchJointPositionsSerializer,
    PyTorchJointVelocitiesSerializer,
    PyTorchJointTorquesSerializer,
    PyTorchCameraFrameSerializer,
    PyTorchAudioFrameSerializer,
    PyTorchIMUSerializer,
    PyTorchTimestampSerializer,
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
