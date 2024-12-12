"""Tests the schema serializer."""

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    CameraFrameSchema,
    DType,
    IMUSchema,
    InputSchema,
    JointPositionsSchema,
    JointPositionUnit,
    JointTorquesSchema,
    JointTorqueUnit,
    JointVelocitiesSchema,
    JointVelocityUnit,
    TimestampSchema,
    ValueSchema,
    VectorCommandSchema,
)
from kinfer.serialize.pytorch import PyTorchInputSerializer
from kinfer.serialize.schema import get_dummy_inputs


def test_serialize_schema() -> None:
    input_schema = InputSchema(
        inputs=[
            ValueSchema(
                value_name="input_1",
                joint_positions=JointPositionsSchema(
                    unit=JointPositionUnit.DEGREES,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            ValueSchema(
                value_name="input_2",
                joint_velocities=JointVelocitiesSchema(
                    unit=JointVelocityUnit.DEGREES_PER_SECOND,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            ValueSchema(
                value_name="input_3",
                joint_torques=JointTorquesSchema(
                    unit=JointTorqueUnit.NEWTON_METERS,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
            ValueSchema(
                value_name="input_4",
                camera_frame=CameraFrameSchema(
                    width=1920,
                    height=1080,
                    channels=3,
                ),
            ),
            ValueSchema(
                value_name="input_5",
                audio_frame=AudioFrameSchema(
                    channels=2,
                    sample_rate=44100,
                    dtype=DType.UINT8,
                ),
            ),
            ValueSchema(
                value_name="input_6",
                imu=IMUSchema(
                    use_accelerometer=True,
                    use_gyroscope=True,
                    use_magnetometer=True,
                ),
            ),
            ValueSchema(
                value_name="input_7",
                timestamp=TimestampSchema(
                    start_seconds=1728000000,
                    start_nanos=0,
                ),
            ),
            ValueSchema(
                value_name="input_8",
                vector_command=VectorCommandSchema(
                    dimensions=3,
                ),
            ),
        ]
    )

    dummy_input = get_dummy_inputs(input_schema)
    serializer = PyTorchInputSerializer(schema=input_schema)
    dummy_input_serialized = serializer.serialize(dummy_input)
    assert len(dummy_input_serialized) == len(input_schema.inputs)
    dummy_input_deserialized = serializer.deserialize(dummy_input_serialized)
    assert len(dummy_input_deserialized.inputs) == len(dummy_input.inputs)
