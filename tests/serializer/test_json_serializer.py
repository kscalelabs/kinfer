"""Tests serialization and deserialization to Numpy arrays."""

import random

import pytest

from kinfer import protos as P
from kinfer.serialize.json import JsonSerializer


@pytest.mark.parametrize("schema_unit", [P.JointPositionUnit.DEGREES, P.JointPositionUnit.RADIANS])
@pytest.mark.parametrize("value_unit", [P.JointPositionUnit.DEGREES, P.JointPositionUnit.RADIANS])
def test_serialize_joint_positions(
    schema_unit: P.JointPositionUnit.ValueType, value_unit: P.JointPositionUnit.ValueType
) -> None:
    serializer = JsonSerializer(
        schema=P.ValueSchema(
            joint_positions=P.JointPositionsSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = P.Value(
        joint_positions=P.JointPositionsValue(
            values=[
                P.JointPositionValue(joint_name="joint_2", value=60, unit=value_unit),
                P.JointPositionValue(joint_name="joint_1", value=30, unit=value_unit),
                P.JointPositionValue(joint_name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert len(new_value.joint_positions.values) == len(value.joint_positions.values)


@pytest.mark.parametrize(
    "schema_unit", [P.JointVelocityUnit.DEGREES_PER_SECOND, P.JointVelocityUnit.RADIANS_PER_SECOND]
)
@pytest.mark.parametrize("value_unit", [P.JointVelocityUnit.DEGREES_PER_SECOND, P.JointVelocityUnit.RADIANS_PER_SECOND])
def test_serialize_joint_velocities(
    schema_unit: P.JointVelocityUnit.ValueType, value_unit: P.JointVelocityUnit.ValueType
) -> None:
    serializer = JsonSerializer(
        schema=P.ValueSchema(
            joint_velocities=P.JointVelocitiesSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = P.Value(
        joint_velocities=P.JointVelocitiesValue(
            values=[
                P.JointVelocityValue(joint_name="joint_2", value=60, unit=value_unit),
                P.JointVelocityValue(joint_name="joint_1", value=30, unit=value_unit),
                P.JointVelocityValue(joint_name="joint_3", value=90, unit=value_unit),
            ]
        )
    )
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert len(new_value.joint_velocities.values) == len(value.joint_velocities.values)


@pytest.mark.parametrize("schema_unit", [P.JointTorqueUnit.NEWTON_METERS])
@pytest.mark.parametrize("value_unit", [P.JointTorqueUnit.NEWTON_METERS])
def test_serialize_joint_torques(
    schema_unit: P.JointTorqueUnit.ValueType, value_unit: P.JointTorqueUnit.ValueType
) -> None:
    serializer = JsonSerializer(
        schema=P.ValueSchema(
            joint_torques=P.JointTorquesSchema(
                unit=schema_unit,
                joint_names=["joint_1", "joint_2", "joint_3"],
            )
        )
    )

    value = P.Value(
        joint_torques=P.JointTorquesValue(
            values=[
                P.JointTorqueValue(joint_name="joint_1", value=1, unit=value_unit),
                P.JointTorqueValue(joint_name="joint_2", value=2, unit=value_unit),
                P.JointTorqueValue(joint_name="joint_3", value=3, unit=value_unit),
            ]
        )
    )
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert len(new_value.joint_torques.values) == len(value.joint_torques.values)


def test_serialize_joint_commands() -> None:
    serializer = JsonSerializer(
        schema=P.ValueSchema(
            joint_commands=P.JointCommandsSchema(
                joint_names=["joint_1", "joint_2", "joint_3"],
                torque_unit=P.JointTorqueUnit.NEWTON_METERS,
                velocity_unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                position_unit=P.JointPositionUnit.RADIANS,
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
                    torque_unit=P.JointTorqueUnit.NEWTON_METERS,
                    velocity_unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                    position_unit=P.JointPositionUnit.RADIANS,
                ),
                P.JointCommandValue(
                    joint_name="joint_2",
                    torque=2,
                    velocity=3,
                    position=4,
                    kp=5,
                    kd=6,
                    torque_unit=P.JointTorqueUnit.NEWTON_METERS,
                    velocity_unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                    position_unit=P.JointPositionUnit.RADIANS,
                ),
                P.JointCommandValue(
                    joint_name="joint_3",
                    torque=3,
                    velocity=4,
                    position=5,
                    kp=6,
                    kd=7,
                    torque_unit=P.JointTorqueUnit.NEWTON_METERS,
                    velocity_unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                    position_unit=P.JointPositionUnit.RADIANS,
                ),
            ]
        )
    )
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    # Back to joint commands value.
    new_value = serializer.deserialize(mapping)
    assert len(new_value.joint_commands.values) == len(value.joint_commands.values)


def test_serialize_camera_frame() -> None:
    serializer = JsonSerializer(
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
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert isinstance(new_value, P.Value)
    assert new_value == value


def test_serialize_audio_frame() -> None:
    serializer = JsonSerializer(
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
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert isinstance(new_value, P.Value)
    assert new_value == value


def test_serialize_imu() -> None:
    serializer = JsonSerializer(
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
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert new_value == value


def test_serialize_timestamp() -> None:
    serializer = JsonSerializer(schema=P.ValueSchema(timestamp=P.TimestampSchema()))

    value = P.Value(
        timestamp=P.TimestampValue(
            seconds=1,
            nanos=500_000_000,
        ),
    )
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)
    assert mapping["seconds"] == 1
    assert mapping["nanos"] == 500_000_000

    new_value = serializer.deserialize(mapping)
    assert new_value == value


def test_serialize_vector_command() -> None:
    serializer = JsonSerializer(schema=P.ValueSchema(vector_command=P.VectorCommandSchema(dimensions=3)))

    value = P.Value(vector_command=P.VectorCommandValue(values=[1.0, 2.0, 3.0]))
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)

    new_value = serializer.deserialize(mapping)
    assert new_value == value


def test_serialize_state_tensor() -> None:
    serializer = JsonSerializer(
        schema=P.ValueSchema(
            state_tensor=P.StateTensorSchema(
                shape=[2, 2],
                dtype=P.DType.INT8,
            )
        )
    )

    value = P.Value(state_tensor=P.StateTensorValue(data=bytes([1, 2, 3, 4])))
    mapping = serializer.serialize(value)
    assert isinstance(mapping, dict)
    assert mapping["data"] == "AQIDBA=="

    new_value = serializer.deserialize(mapping)
    assert new_value == value
