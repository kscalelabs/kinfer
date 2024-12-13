"""Tests serialization and deserialization to PyTorch tensors."""

import random

import pytest
from torch import Tensor

from kinfer import protos as P
from kinfer.serialize.pytorch import PyTorchSerializer
from kinfer.serialize.types import to_value_type


@pytest.mark.parametrize("schema_unit", [P.JointPositionUnit.DEGREES, P.JointPositionUnit.RADIANS])
@pytest.mark.parametrize("value_unit", [P.JointPositionUnit.DEGREES, P.JointPositionUnit.RADIANS])
def test_serialize_joint_positions(schema_unit: P.JointPositionUnit, value_unit: P.JointPositionUnit) -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            joint_positions=P.JointPositionsSchema(
                unit=to_value_type(schema_unit),
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    # From joint positions to tensor.
    value = P.Value(
        joint_positions=P.JointPositionsValue(
            values=[
                P.JointPositionValue(joint_name="joint_2", value=60, unit=to_value_type(value_unit)),
                P.JointPositionValue(joint_name="joint_1", value=30, unit=to_value_type(value_unit)),
                P.JointPositionValue(joint_name="joint_3", value=90, unit=to_value_type(value_unit)),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint positions value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_positions.values) == len(value.joint_positions.values)


@pytest.mark.parametrize(
    "schema_unit", [P.JointVelocityUnit.DEGREES_PER_SECOND, P.JointVelocityUnit.RADIANS_PER_SECOND]
)
@pytest.mark.parametrize("value_unit", [P.JointVelocityUnit.DEGREES_PER_SECOND, P.JointVelocityUnit.RADIANS_PER_SECOND])
def test_serialize_joint_velocities(schema_unit: P.JointVelocityUnit, value_unit: P.JointVelocityUnit) -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            joint_velocities=P.JointVelocitiesSchema(
                unit=to_value_type(schema_unit),
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = P.Value(
        joint_velocities=P.JointVelocitiesValue(
            values=[
                P.JointVelocityValue(joint_name="joint_2", value=60, unit=to_value_type(value_unit)),
                P.JointVelocityValue(joint_name="joint_1", value=30, unit=to_value_type(value_unit)),
                P.JointVelocityValue(joint_name="joint_3", value=90, unit=to_value_type(value_unit)),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint velocities value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_velocities.values) == len(value.joint_velocities.values)


@pytest.mark.parametrize("schema_unit", [P.JointTorqueUnit.NEWTON_METERS])
@pytest.mark.parametrize("value_unit", [P.JointTorqueUnit.NEWTON_METERS])
def test_serialize_joint_torques(schema_unit: P.JointTorqueUnit, value_unit: P.JointTorqueUnit) -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            joint_torques=P.JointTorquesSchema(
                unit=to_value_type(schema_unit),
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = P.Value(
        joint_torques=P.JointTorquesValue(
            values=[
                P.JointTorqueValue(joint_name="joint_1", value=1, unit=to_value_type(value_unit)),
                P.JointTorqueValue(joint_name="joint_2", value=2, unit=to_value_type(value_unit)),
                P.JointTorqueValue(joint_name="joint_3", value=3, unit=to_value_type(value_unit)),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint torques value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_torques.values) == len(value.joint_torques.values)


def test_serialize_joint_commands() -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            joint_commands=P.JointCommandsSchema(
                joint_names=["joint_1", "joint_2", "joint_3"],
                torque_unit=to_value_type(P.JointTorqueUnit.NEWTON_METERS),
                velocity_unit=to_value_type(P.JointVelocityUnit.RADIANS_PER_SECOND),
                position_unit=to_value_type(P.JointPositionUnit.RADIANS),
            )
        )
    )

    value = P.Value(
        joint_commands=P.JointCommandsValue(
            values=[
                P.JointCommandValue(
                    joint_name="joint_1",
                    torque=1,
                    velocity=2,
                    position=3,
                    kp=4,
                    kd=5,
                    torque_unit=to_value_type(P.JointTorqueUnit.NEWTON_METERS),
                    velocity_unit=to_value_type(P.JointVelocityUnit.RADIANS_PER_SECOND),
                    position_unit=to_value_type(P.JointPositionUnit.RADIANS),
                ),
                P.JointCommandValue(
                    joint_name="joint_2",
                    torque=2,
                    velocity=3,
                    position=4,
                    kp=5,
                    kd=6,
                    torque_unit=to_value_type(P.JointTorqueUnit.NEWTON_METERS),
                    velocity_unit=to_value_type(P.JointVelocityUnit.RADIANS_PER_SECOND),
                    position_unit=to_value_type(P.JointPositionUnit.RADIANS),
                ),
                P.JointCommandValue(
                    joint_name="joint_3",
                    torque=3,
                    velocity=4,
                    position=5,
                    kp=6,
                    kd=7,
                    torque_unit=to_value_type(P.JointTorqueUnit.NEWTON_METERS),
                    velocity_unit=to_value_type(P.JointVelocityUnit.RADIANS_PER_SECOND),
                    position_unit=to_value_type(P.JointPositionUnit.RADIANS),
                ),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint commands value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_commands.values) == len(value.joint_commands.values)


def test_serialize_camera_frame() -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            camera_frame=P.CameraFrameSchema(
                width=32,
                height=64,
                channels=3,
            )
        )
    )

    value = P.Value(
        camera_frame=P.CameraFrameValue(
            data=bytes([random.randint(0, 255) for _ in range(32 * 64 * 3)]),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (3, 64, 32)

    # Back to camera frame value.
    new_value = serializer.deserialize(tensor)
    assert isinstance(new_value, P.Value)
    assert new_value == value


def test_serialize_audio_frame() -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            audio_frame=P.AudioFrameSchema(
                channels=2,
                sample_rate=44100,
                dtype=P.DType.UINT16,
            )
        )
    )

    value = P.Value(
        audio_frame=P.AudioFrameValue(
            data=bytes([random.randint(0, 255) for _ in range(44100 * 2 * 2)]),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to audio frame value.
    new_value = serializer.deserialize(tensor)
    assert isinstance(new_value, P.Value)
    assert new_value == value


def test_serialize_imu() -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            imu=P.IMUSchema(
                use_accelerometer=True,
                use_gyroscope=True,
                use_magnetometer=True,
            )
        )
    )

    value = P.Value(
        imu=P.IMUValue(
            linear_acceleration=P.IMUAccelerometerValue(x=1.0, y=2.0, z=3.0),
            angular_velocity=P.IMUGyroscopeValue(x=4.0, y=5.0, z=6.0),
            magnetic_field=P.IMUMagnetometerValue(x=7.0, y=8.0, z=9.0),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to imu value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_timestamp() -> None:
    serializer = PyTorchSerializer(schema=P.ValueSchema(timestamp=P.TimestampSchema()))

    # From timestamp value to tensor.
    value = P.Value(
        timestamp=P.TimestampValue(
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
    serializer = PyTorchSerializer(schema=P.ValueSchema(vector_command=P.VectorCommandSchema()))

    value = P.Value(vector_command=P.VectorCommandValue(values=[1.0, 2.0, 3.0]))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (3,)

    # Back to vector command value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_state_tensor() -> None:
    serializer = PyTorchSerializer(
        schema=P.ValueSchema(
            state_tensor=P.StateTensorSchema(
                shape=[2, 2],
                dtype=P.DType.INT8,
            )
        )
    )

    value = P.Value(state_tensor=P.StateTensorValue(data=bytes([1, 2, 3, 4])))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (2, 2)

    # Back to state tensor value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value
