"""Tests the schema serializer."""

from kinfer import proto as P
from kinfer.serialize.pytorch import PyTorchMultiSerializer
from kinfer.serialize.schema import get_dummy_io


def test_serialize_schema() -> None:
    input_schema = P.IOSchema(
        values=[
            P.ValueSchema(
                value_name="input_1",
                joint_positions=P.JointPositionsSchema(
                    unit=P.JointPositionUnit.DEGREES,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            P.ValueSchema(
                value_name="input_2",
                joint_velocities=P.JointVelocitiesSchema(
                    unit=P.JointVelocityUnit.DEGREES_PER_SECOND,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            P.ValueSchema(
                value_name="input_3",
                joint_torques=P.JointTorquesSchema(
                    unit=P.JointTorqueUnit.NEWTON_METERS,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            P.ValueSchema(
                value_name="input_4",
                camera_frame=P.CameraFrameSchema(
                    width=1920,
                    height=1080,
                    channels=3,
                ),
            ),
            P.ValueSchema(
                value_name="input_5",
                audio_frame=P.AudioFrameSchema(
                    channels=2,
                    sample_rate=44100,
                    dtype=P.DType.UINT8,
                ),
            ),
            P.ValueSchema(
                value_name="input_6",
                imu=P.IMUSchema(
                    use_accelerometer=True,
                    use_gyroscope=True,
                    use_magnetometer=True,
                ),
            ),
            P.ValueSchema(
                value_name="input_7",
                timestamp=P.TimestampSchema(
                    start_seconds=1728000000,
                    start_nanos=0,
                ),
            ),
            P.ValueSchema(
                value_name="input_8",
                vector_command=P.VectorCommandSchema(
                    dimensions=3,
                ),
            ),
        ]
    )

    dummy_input = get_dummy_io(input_schema)
    serializer = PyTorchMultiSerializer(schema=input_schema)
    dummy_input_serialized = serializer.serialize_io(dummy_input)
    assert len(dummy_input_serialized) == len(input_schema.values)
    dummy_input_deserialized = serializer.deserialize_io(dummy_input_serialized)
    assert len(dummy_input_deserialized.values) == len(dummy_input.values)
