"""Tests the schema serializer."""

from kinfer import proto as K
from kinfer.serialize.pytorch import PyTorchMultiSerializer
from kinfer.serialize.schema import get_dummy_io


def test_serialize_schema() -> None:
    input_schema = K.IOSchema(
        values=[
            K.ValueSchema(
                value_name="input_1",
                joint_positions=K.JointPositionsSchema(
                    unit=K.JointPositionUnit.DEGREES,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            K.ValueSchema(
                value_name="input_2",
                joint_velocities=K.JointVelocitiesSchema(
                    unit=K.JointVelocityUnit.DEGREES_PER_SECOND,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            K.ValueSchema(
                value_name="input_3",
                joint_torques=K.JointTorquesSchema(
                    unit=K.JointTorqueUnit.NEWTON_METERS,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            K.ValueSchema(
                value_name="input_4",
                camera_frame=K.CameraFrameSchema(
                    width=1920,
                    height=1080,
                    channels=3,
                ),
            ),
            K.ValueSchema(
                value_name="input_5",
                audio_frame=K.AudioFrameSchema(
                    channels=2,
                    sample_rate=44100,
                    dtype=K.DType.UINT8,
                ),
            ),
            K.ValueSchema(
                value_name="input_6",
                imu=K.ImuSchema(
                    use_accelerometer=True,
                    use_gyroscope=True,
                    use_magnetometer=True,
                ),
            ),
            K.ValueSchema(
                value_name="input_7",
                timestamp=K.TimestampSchema(
                    start_seconds=1728000000,
                    start_nanos=0,
                ),
            ),
            K.ValueSchema(
                value_name="input_8",
                vector_command=K.VectorCommandSchema(
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
