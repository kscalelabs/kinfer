"""Tests serialization and deserialization to PyTorch tensors."""

import random

import pytest
from torch import Tensor

from kinfer import proto as K
from kinfer.serialize.pytorch import PyTorchSerializer
from kinfer.serialize.types import to_value_type


@pytest.mark.parametrize("schema_unit", [K.JointPositionUnit.DEGREES, K.JointPositionUnit.RADIANS])
@pytest.mark.parametrize("value_unit", [K.JointPositionUnit.DEGREES, K.JointPositionUnit.RADIANS])
def test_serialize_joint_positions(schema_unit: K.JointPositionUnit, value_unit: K.JointPositionUnit) -> None:
    serializer = PyTorchSerializer(
        schema=K.ValueSchema(
            joint_positions=K.JointPositionsSchema(
                unit=to_value_type(schema_unit),
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    # From joint positions to tensor.
    value = K.Value(
        joint_positions=K.JointPositionsValue(
            values=[
                K.JointPositionValue(joint_name="joint_2", value=60, unit=to_value_type(value_unit)),
                K.JointPositionValue(joint_name="joint_1", value=30, unit=to_value_type(value_unit)),
                K.JointPositionValue(joint_name="joint_3", value=90, unit=to_value_type(value_unit)),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint positions value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_positions.values) == len(value.joint_positions.values)


@pytest.mark.parametrize(
    "schema_unit", [K.JointVelocityUnit.DEGREES_PER_SECOND, K.JointVelocityUnit.RADIANS_PER_SECOND]
)
@pytest.mark.parametrize("value_unit", [K.JointVelocityUnit.DEGREES_PER_SECOND, K.JointVelocityUnit.RADIANS_PER_SECOND])
def test_serialize_joint_velocities(schema_unit: K.JointVelocityUnit, value_unit: K.JointVelocityUnit) -> None:
    serializer = PyTorchSerializer(
        schema=K.ValueSchema(
            joint_velocities=K.JointVelocitiesSchema(
                unit=to_value_type(schema_unit),
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = K.Value(
        joint_velocities=K.JointVelocitiesValue(
            values=[
                K.JointVelocityValue(joint_name="joint_2", value=60, unit=to_value_type(value_unit)),
                K.JointVelocityValue(joint_name="joint_1", value=30, unit=to_value_type(value_unit)),
                K.JointVelocityValue(joint_name="joint_3", value=90, unit=to_value_type(value_unit)),
            ]
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to joint velocities value.
    new_value = serializer.deserialize(tensor)
    assert len(new_value.joint_velocities.values) == len(value.joint_velocities.values)


@pytest.mark.parametrize("schema_unit", [K.JointTorqueUnit.NEWTON_METERS])
@pytest.mark.parametrize("value_unit", [K.JointTorqueUnit.NEWTON_METERS])
def test_serialize_joint_torques(schema_unit: K.JointTorqueUnit, value_unit: K.JointTorqueUnit) -> None:
    serializer = PyTorchSerializer(
        schema=K.ValueSchema(
            joint_torques=K.JointTorquesSchema(
                unit=to_value_type(schema_unit),
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = K.Value(
        joint_torques=K.JointTorquesValue(
            values=[
                K.JointTorqueValue(joint_name="joint_1", value=1, unit=to_value_type(value_unit)),
                K.JointTorqueValue(joint_name="joint_2", value=2, unit=to_value_type(value_unit)),
                K.JointTorqueValue(joint_name="joint_3", value=3, unit=to_value_type(value_unit)),
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
        schema=K.ValueSchema(
            joint_commands=K.JointCommandsSchema(
                joint_names=["joint_1", "joint_2", "joint_3"],
                torque_unit=to_value_type(K.JointTorqueUnit.NEWTON_METERS),
                velocity_unit=to_value_type(K.JointVelocityUnit.RADIANS_PER_SECOND),
                position_unit=to_value_type(K.JointPositionUnit.RADIANS),
            )
        )
    )

    value = K.Value(
        joint_commands=K.JointCommandsValue(
            values=[
                K.JointCommandValue(
                    joint_name="joint_1",
                    torque=1,
                    velocity=2,
                    position=3,
                    kp=4,
                    kd=5,
                    torque_unit=to_value_type(K.JointTorqueUnit.NEWTON_METERS),
                    velocity_unit=to_value_type(K.JointVelocityUnit.RADIANS_PER_SECOND),
                    position_unit=to_value_type(K.JointPositionUnit.RADIANS),
                ),
                K.JointCommandValue(
                    joint_name="joint_2",
                    torque=2,
                    velocity=3,
                    position=4,
                    kp=5,
                    kd=6,
                    torque_unit=to_value_type(K.JointTorqueUnit.NEWTON_METERS),
                    velocity_unit=to_value_type(K.JointVelocityUnit.RADIANS_PER_SECOND),
                    position_unit=to_value_type(K.JointPositionUnit.RADIANS),
                ),
                K.JointCommandValue(
                    joint_name="joint_3",
                    torque=3,
                    velocity=4,
                    position=5,
                    kp=6,
                    kd=7,
                    torque_unit=to_value_type(K.JointTorqueUnit.NEWTON_METERS),
                    velocity_unit=to_value_type(K.JointVelocityUnit.RADIANS_PER_SECOND),
                    position_unit=to_value_type(K.JointPositionUnit.RADIANS),
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
        schema=K.ValueSchema(
            camera_frame=K.CameraFrameSchema(
                width=32,
                height=64,
                channels=3,
            )
        )
    )

    value = K.Value(
        camera_frame=K.CameraFrameValue(
            data=bytes([random.randint(0, 255) for _ in range(32 * 64 * 3)]),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (3, 64, 32)

    # Back to camera frame value.
    new_value = serializer.deserialize(tensor)
    assert isinstance(new_value, K.Value)
    assert new_value == value


def test_serialize_audio_frame() -> None:
    serializer = PyTorchSerializer(
        schema=K.ValueSchema(
            audio_frame=K.AudioFrameSchema(
                channels=2,
                sample_rate=44100,
                dtype=K.DType.UINT16,
            )
        )
    )

    value = K.Value(
        audio_frame=K.AudioFrameValue(
            data=bytes([random.randint(0, 255) for _ in range(44100 * 2 * 2)]),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to audio frame value.
    new_value = serializer.deserialize(tensor)
    assert isinstance(new_value, K.Value)
    assert new_value == value


def test_serialize_imu() -> None:
    serializer = PyTorchSerializer(
        schema=K.ValueSchema(
            imu=K.ImuSchema(
                use_accelerometer=True,
                use_gyroscope=True,
                use_magnetometer=True,
            )
        )
    )

    value = K.Value(
        imu=K.ImuValue(
            linear_acceleration=K.ImuAccelerometerValue(x=1.0, y=2.0, z=3.0),
            angular_velocity=K.ImuGyroscopeValue(x=4.0, y=5.0, z=6.0),
            magnetic_field=K.ImuMagnetometerValue(x=7.0, y=8.0, z=9.0),
        )
    )
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)

    # Back to imu value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_timestamp() -> None:
    serializer = PyTorchSerializer(schema=K.ValueSchema(timestamp=K.TimestampSchema()))

    # From timestamp value to tensor.
    value = K.Value(
        timestamp=K.TimestampValue(
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
    serializer = PyTorchSerializer(schema=K.ValueSchema(vector_command=K.VectorCommandSchema(dimensions=3)))

    value = K.Value(vector_command=K.VectorCommandValue(values=[1.0, 2.0, 3.0]))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (3,)

    # Back to vector command value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value


def test_serialize_state_tensor() -> None:
    serializer = PyTorchSerializer(
        schema=K.ValueSchema(
            state_tensor=K.StateTensorSchema(
                shape=[2, 2],
                dtype=K.DType.INT8,
            )
        )
    )

    value = K.Value(state_tensor=K.StateTensorValue(data=bytes([1, 2, 3, 4])))
    tensor = serializer.serialize(value)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (2, 2)

    # Back to state tensor value.
    new_value = serializer.deserialize(tensor)
    assert new_value == value
