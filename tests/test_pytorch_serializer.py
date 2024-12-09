"""Tests serialization and deserialization to PyTorch tensors."""

import random

import pytest
from torch import Tensor

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    AudioFrameValue,
    CameraFrameSchema,
    CameraFrameValue,
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
    Value,
    ValueSchema,
)
from kinfer.serialize.pytorch import PyTorchSerializer


def test_serialize_tensor() -> None:
    serializer = PyTorchSerializer(schema=ValueSchema(tensor=TensorSchema(shape=[1, 2, 3])))

    # From tensor value to tensor.
    value = Value(tensor=TensorValue(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (1, 2, 3)

    # Back to tensor value.
    new_value = serializer.deserialize("tensor", tensor)
    assert new_value == value


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
                JointPositionValue(name="joint_2", value=60, unit=value_unit),
                JointPositionValue(name="joint_1", value=30, unit=value_unit),
                JointPositionValue(name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint positions value.
    new_value = serializer.deserialize("joint_positions", tensor)
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
                JointVelocityValue(name="joint_2", value=60, unit=value_unit),
                JointVelocityValue(name="joint_1", value=30, unit=value_unit),
                JointVelocityValue(name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint velocities value.
    new_value = serializer.deserialize("joint_velocities", tensor)
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
                JointTorqueValue(name="joint_1", value=1, unit=value_unit),
                JointTorqueValue(name="joint_2", value=2, unit=value_unit),
                JointTorqueValue(name="joint_3", value=3, unit=value_unit),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint torques value.
    new_value = serializer.deserialize("joint_torques", tensor)
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
    new_value = serializer.deserialize("camera_frame", tensor)
    assert isinstance(new_value, Value)
    assert new_value == value


def test_serialize_audio_frame() -> None:
    serializer = PyTorchSerializer(
        schema=ValueSchema(
            audio_frame=AudioFrameSchema(
                channels=2,
                sample_rate=44100,
                bytes_per_sample=2,
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
    new_value = serializer.deserialize("audio_frame", tensor)
    assert isinstance(new_value, Value)
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
    new_value = serializer.deserialize("timestamp", tensor)
    assert new_value == value
