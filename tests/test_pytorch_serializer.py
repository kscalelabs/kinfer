"""Tests serialization and deserialization to PyTorch tensors."""

import random

import pytest
from torch import Tensor

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    AudioFrameValue,
    CameraFrameSchema,
    CameraFrameValue,
    DType,
    IMUAccelerometerValue,
    IMUGyroscopeValue,
    IMUMagnetometerValue,
    IMUSchema,
    IMUValue,
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
    StateTensorSchema,
    StateTensorValue,
    TimestampSchema,
    TimestampValue,
    Value,
    ValueSchema,
    VectorCommandSchema,
    VectorCommandValue,
)
from kinfer.serialize.pytorch import PyTorchSerializer


@pytest.mark.parametrize("schema_unit", [JointPositionUnit.DEGREES, JointPositionUnit.RADIANS])
@pytest.mark.parametrize("value_unit", [JointPositionUnit.DEGREES, JointPositionUnit.RADIANS])
def test_serialize_joint_positions(schema_unit: JointPositionUnit, value_unit: JointPositionUnit) -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            joint_positions=JointPositionsSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    # From joint positions to tensor.
    value = Value(
        joint_positions=JointPositionsValue(
            values=[
                JointPositionValue(joint_name="joint_2", value=60, unit=value_unit),
                JointPositionValue(joint_name="joint_1", value=30, unit=value_unit),
                JointPositionValue(joint_name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint positions value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_positions.values) == len(value.joint_positions.values)


@pytest.mark.parametrize("schema_unit", [JointVelocityUnit.DEGREES_PER_SECOND, JointVelocityUnit.RADIANS_PER_SECOND])
@pytest.mark.parametrize("value_unit", [JointVelocityUnit.DEGREES_PER_SECOND, JointVelocityUnit.RADIANS_PER_SECOND])
def test_serialize_joint_velocities(schema_unit: JointVelocityUnit, value_unit: JointVelocityUnit) -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            joint_velocities=JointVelocitiesSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = Value(
        joint_velocities=JointVelocitiesValue(
            values=[
                JointVelocityValue(joint_name="joint_2", value=60, unit=value_unit),
                JointVelocityValue(joint_name="joint_1", value=30, unit=value_unit),
                JointVelocityValue(joint_name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint velocities value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_velocities.values) == len(value.joint_velocities.values)


@pytest.mark.parametrize("schema_unit", [JointTorqueUnit.NEWTON_METERS])
@pytest.mark.parametrize("value_unit", [JointTorqueUnit.NEWTON_METERS])
def test_serialize_joint_torques(schema_unit: JointTorqueUnit, value_unit: JointTorqueUnit) -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            joint_torques=JointTorquesSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = Value(
        joint_torques=JointTorquesValue(
            values=[
                JointTorqueValue(joint_name="joint_1", value=1, unit=value_unit),
                JointTorqueValue(joint_name="joint_2", value=2, unit=value_unit),
                JointTorqueValue(joint_name="joint_3", value=3, unit=value_unit),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint torques value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_torques.values) == len(value.joint_torques.values)


def test_serialize_camera_frame() -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            camera_frame=CameraFrameSchema(
                width=32,
                height=64,
                channels=3,
            )
        )
    )

    value = Value(
        camera_frame=CameraFrameValue(
            data=bytes([random.randint(0, 255) for _ in range(32 * 64 * 3)]),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (3, 64, 32)

    # Back to camera frame value.
    new_value = serializer.deserialize(tensor)
    assert isinstance(new_value, Value)
    assert new_value == value


def test_serialize_audio_frame() -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            audio_frame=AudioFrameSchema(
                channels=2,
                sample_rate=44100,
                dtype=DType.UINT16,
            )
        )
    )

    value = Value(
        audio_frame=AudioFrameValue(
            data=bytes([random.randint(0, 255) for _ in range(44100 * 2 * 2)]),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to audio frame value.
    new_value = serializer.deserialize(tensor)
    assert isinstance(new_value, Value)
    assert new_value == value


def test_serialize_imu() -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            imu=IMUSchema(
                use_accelerometer=True,
                use_gyroscope=True,
                use_magnetometer=True,
            )
        )
    )

    value = Value(
        imu=IMUValue(
            linear_acceleration=IMUAccelerometerValue(x=1.0, y=2.0, z=3.0),
            angular_velocity=IMUGyroscopeValue(x=4.0, y=5.0, z=6.0),
            magnetic_field=IMUMagnetometerValue(x=7.0, y=8.0, z=9.0),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to imu value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_timestamp() -> None:
    serializer = PyTorchSerializer(schema=ValueSchema(timestamp=TimestampSchema()))

    # From timestamp value to tensor.
    value = Value(
        timestamp=TimestampValue(
            seconds=1,
            nanos=500_000_000,
        ),
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.item() == 1.5

    # Back to timestamp value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_vector_command() -> None:
    serializer = PyTorchSerializer(schema=ValueSchema(vector_command=VectorCommandSchema()))

    value = Value(vector_command=VectorCommandValue(values=[1.0, 2.0, 3.0]))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (3,)

    # Back to vector command value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_state_tensor() -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            state_tensor=StateTensorSchema(
                shape=[2, 2],
                dtype=DType.INT8,
            )
        )
    )

    value = Value(state_tensor=StateTensorValue(data=bytes([1, 2, 3, 4])))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (2, 2)

    # Back to state tensor value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value
