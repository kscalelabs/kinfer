"""Tests serialization and deserialization to Numpy arrays."""

import random

import numpy as np
import pytest

from kinfer import proto as K
from kinfer.serialize.numpy import NumpySerializer


@pytest.mark.parametrize("schema_unit", [K.JointPositionUnit.DEGREES, K.JointPositionUnit.RADIANS])
@pytest.mark.parametrize("value_unit", [K.JointPositionUnit.DEGREES, K.JointPositionUnit.RADIANS])
def test_serialize_joint_positions(
    schema_unit: K.JointPositionUnit.ValueType,
    value_unit: K.JointPositionUnit.ValueType,
) -> None:
    serializer = NumpySerializer(
        schema=K.ValueSchema(
            joint_positions=K.JointPositionsSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = K.Value(
        joint_positions=K.JointPositionsValue(
            values=[
                K.JointPositionValue(joint_name="joint_2", value=60, unit=value_unit),
                K.JointPositionValue(joint_name="joint_1", value=30, unit=value_unit),
                K.JointPositionValue(joint_name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)

    new_value = serializer.deserialize(array)
    assert len(new_value.joint_positions.values) == len(value.joint_positions.values)


@pytest.mark.parametrize(
    "schema_unit", [K.JointVelocityUnit.DEGREES_PER_SECOND, K.JointVelocityUnit.RADIANS_PER_SECOND]
)
@pytest.mark.parametrize("value_unit", [K.JointVelocityUnit.DEGREES_PER_SECOND, K.JointVelocityUnit.RADIANS_PER_SECOND])
def test_serialize_joint_velocities(
    schema_unit: K.JointVelocityUnit.ValueType,
    value_unit: K.JointVelocityUnit.ValueType,
) -> None:
    serializer = NumpySerializer(
        schema=K.ValueSchema(
            joint_velocities=K.JointVelocitiesSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = K.Value(
        joint_velocities=K.JointVelocitiesValue(
            values=[
                K.JointVelocityValue(joint_name="joint_2", value=60, unit=value_unit),
                K.JointVelocityValue(joint_name="joint_1", value=30, unit=value_unit),
                K.JointVelocityValue(joint_name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)

    new_value = serializer.deserialize(array)
    assert len(new_value.joint_velocities.values) == len(value.joint_velocities.values)


@pytest.mark.parametrize("schema_unit", [K.JointTorqueUnit.NEWTON_METERS])
@pytest.mark.parametrize("value_unit", [K.JointTorqueUnit.NEWTON_METERS])
def test_serialize_joint_torques(
    schema_unit: K.JointTorqueUnit.ValueType,
    value_unit: K.JointTorqueUnit.ValueType,
) -> None:
    serializer = NumpySerializer(
        schema=K.ValueSchema(
            joint_torques=K.JointTorquesSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = K.Value(
        joint_torques=K.JointTorquesValue(
            values=[
                K.JointTorqueValue(joint_name="joint_1", value=1, unit=value_unit),
                K.JointTorqueValue(joint_name="joint_2", value=2, unit=value_unit),
                K.JointTorqueValue(joint_name="joint_3", value=3, unit=value_unit),
            ]
        )
    )
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)

    new_value = serializer.deserialize(array)
    assert len(new_value.joint_torques.values) == len(value.joint_torques.values)


def test_serialize_joint_commands() -> None:
    serializer = NumpySerializer(
        schema=K.ValueSchema(
            joint_commands=K.JointCommandsSchema(
                joint_names=["joint_1", "joint_2", "joint_3"],
                torque_unit=K.JointTorqueUnit.NEWTON_METERS,
                velocity_unit=K.JointVelocityUnit.RADIANS_PER_SECOND,
                position_unit=K.JointPositionUnit.RADIANS,
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
                    torque_unit=K.JointTorqueUnit.NEWTON_METERS,
                    velocity_unit=K.JointVelocityUnit.RADIANS_PER_SECOND,
                    position_unit=K.JointPositionUnit.RADIANS,
                ),
                K.JointCommandValue(
                    joint_name="joint_2",
                    torque=2,
                    velocity=3,
                    position=4,
                    kp=5,
                    kd=6,
                    torque_unit=K.JointTorqueUnit.NEWTON_METERS,
                    velocity_unit=K.JointVelocityUnit.RADIANS_PER_SECOND,
                    position_unit=K.JointPositionUnit.RADIANS,
                ),
                K.JointCommandValue(
                    joint_name="joint_3",
                    torque=3,
                    velocity=4,
                    position=5,
                    kp=6,
                    kd=7,
                    torque_unit=K.JointTorqueUnit.NEWTON_METERS,
                    velocity_unit=K.JointVelocityUnit.RADIANS_PER_SECOND,
                    position_unit=K.JointPositionUnit.RADIANS,
                ),
            ]
        )
    )
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)

    # Back to joint commands value.
    new_value = serializer.deserialize(array)
    assert len(new_value.joint_commands.values) == len(value.joint_commands.values)


def test_serialize_camera_frame() -> None:
    serializer = NumpySerializer(
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
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 64, 32)

    new_value = serializer.deserialize(array)
    assert isinstance(new_value, K.Value)
    assert new_value == value


def test_serialize_audio_frame() -> None:
    serializer = NumpySerializer(
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
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)

    new_value = serializer.deserialize(array)
    assert isinstance(new_value, K.Value)
    assert new_value == value


def test_serialize_imu() -> None:
    serializer = NumpySerializer(
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
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)

    new_value = serializer.deserialize(array)
    assert new_value == value


def test_serialize_timestamp() -> None:
    serializer = NumpySerializer(schema=K.ValueSchema(timestamp=K.TimestampSchema()))

    value = K.Value(
        timestamp=K.TimestampValue(
            seconds=1,
            nanos=500_000_000,
        ),
    )
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)
    assert float(array.item()) == 1.5

    new_value = serializer.deserialize(array)
    assert new_value == value


def test_serialize_vector_command() -> None:
    serializer = NumpySerializer(schema=K.ValueSchema(vector_command=K.VectorCommandSchema(dimensions=3)))

    value = K.Value(vector_command=K.VectorCommandValue(values=[1.0, 2.0, 3.0]))
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)
    assert array.shape == (3,)

    new_value = serializer.deserialize(array)
    assert new_value == value


def test_serialize_state_tensor() -> None:
    serializer = NumpySerializer(
        schema=K.ValueSchema(
            state_tensor=K.StateTensorSchema(
                shape=[2, 2],
                dtype=K.DType.INT8,
            )
        )
    )

    value = K.Value(state_tensor=K.StateTensorValue(data=bytes([1, 2, 3, 4])))
    array = serializer.serialize(value)
    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 2)

    new_value = serializer.deserialize(array)
    assert new_value == value
