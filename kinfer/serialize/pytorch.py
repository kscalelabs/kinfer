"""Defines a serializer for PyTorch tensors."""

import math
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
    OutputSchema,
    StateTensorSchema,
    StateTensorValue,
    TimestampSchema,
    TimestampValue,
    ValueSchema,
    VectorCommandSchema,
    VectorCommandValue,
    AngularVelocitySchema,
    AngularVelocityValue,
    EulerRotationSchema,
    EulerRotationValue,
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
    StateTensorSerializer,
    TimestampSerializer,
    VectorCommandSerializer,
    AngularVelocitySerializer,
    EulerRotationSerializer,
)
from kinfer.serialize.utils import dtype_num_bytes, dtype_range, numpy_dtype, parse_bytes


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
        # elapsed_nanos = value.nanos - schema.start_nanos
        # if elapsed_nanos < 0:
        #     elapsed_seconds -= 1
        #     elapsed_nanos += 1_000_000_000
        # total_elapsed_seconds = elapsed_seconds + elapsed_nanos / 1_000_000_000
        return torch.tensor([value.seconds], dtype=self.dtype, device=self.device, requires_grad=False)

    def deserialize_timestamp(self, schema: TimestampSchema, value: Tensor) -> TimestampValue:
        total_elapsed_seconds = value.item()
        elapsed_seconds = int(total_elapsed_seconds)
        # elapsed_nanos = int((total_elapsed_seconds - elapsed_seconds) * 1_000_000_000)
        
        return TimestampValue(seconds=elapsed_seconds) #, nanos=elapsed_nanos)


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
        tensor = torch.from_numpy(np_arr).to(self.device, self.dtype)
        tensor = tensor.view(tuple(schema.shape))
        return tensor

    def deserialize_state_tensor(self, schema: StateTensorSchema, value: Tensor) -> StateTensorValue:
        return StateTensorValue(data=value.cpu().flatten().numpy().tobytes())


class PyTorchAngularVelocitySerializer(PyTorchBaseSerializer, AngularVelocitySerializer[Tensor]):
    def serialize_angular_velocity(self, schema: AngularVelocitySchema, value: AngularVelocityValue) -> Tensor:
        return torch.tensor(
            [value.x, value.y, value.z],
            dtype=self.dtype,
            device=self.device,
        )

    def deserialize_angular_velocity(self, schema: AngularVelocitySchema, value: Tensor) -> AngularVelocityValue:
        x, y, z = value.tolist()
        return AngularVelocityValue(x=x, y=y, z=z)


class PyTorchEulerRotationSerializer(PyTorchBaseSerializer, EulerRotationSerializer[Tensor]):
    def serialize_euler_rotation(self, schema: EulerRotationSchema, value: EulerRotationValue) -> Tensor:
        return torch.tensor(
            [value.x, value.y, value.z],
            dtype=self.dtype,
            device=self.device,
        )

    def deserialize_euler_rotation(self, schema: EulerRotationSchema, value: Tensor) -> EulerRotationValue:
        x, y, z = value.tolist()
        return EulerRotationValue(x=x, y=y, z=z)


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
    PyTorchAngularVelocitySerializer,
    PyTorchEulerRotationSerializer,
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


class PyTorchOutputSerializer(MultiSerializer[Tensor]):
    def __init__(self, schema: OutputSchema) -> None:
        super().__init__([PyTorchSerializer(schema=s) for s in schema.outputs])


if __name__ == "__main__":
    # Create an example InputSchema with multiple value types
    schema = InputSchema(inputs=[
        ValueSchema(
            joint_positions=JointPositionsSchema(
                joint_names=["joint1", "joint2"],
                unit=JointPositionUnit.RADIANS
            )
        ),
        ValueSchema(
            timestamp=TimestampSchema(
                start_seconds=0,
                start_nanos=0
            )
        )
    ])

    # Create example input values
    joint_positions = JointPositionsValue(values=[
        JointPositionValue(joint_name="joint1", value=1.5, unit=JointPositionUnit.RADIANS),
        JointPositionValue(joint_name="joint2", value=0.5, unit=JointPositionUnit.RADIANS)
    ])
    timestamp = TimestampValue(seconds=1, nanos=500000000)  # 1.5 seconds

    # Create and use the serializer
    serializer = PyTorchInputSerializer(schema)
    tensors = serializer.serialize(timestamp)

    print("Serialized tensors:", tensors)
    # Expected output will be a list of two tensors:
    # [tensor([1.5000, 0.5000]), tensor([1.5000])]

    # Deserialize back
    deserialized = serializer.deserialize(tensors)
    print("Deserialized values:", deserialized)
