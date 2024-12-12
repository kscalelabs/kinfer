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
    Output,
    OutputSchema,
    TimestampSchema,
    ValueSchema,
    
    VectorCommandSchema,
)
from kinfer.serialize.pytorch import PyTorchInputSerializer, PyTorchOutputSerializer
from kinfer.serialize.schema import get_dummy_inputs, get_dummy_outputs


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
    dummy_input_d = serializer.(dummy_input_serialized)
    assert len(dummy_input_d.inputs) == len(dummy_input.inputs)


def test_serialize_output_schema() -> None:
    output_schema = OutputSchema(
        outputs=[
            ValueSchema(
                value_name="output_1",
                joint_positions=JointPositionsSchema(
                    unit=JointPositionUnit.DEGREES,
                    joint_names=["joint_1", "joint_2", "joint_3"],
                ),
            ),
        ]
    )
    dummy_output = get_dummy_outputs(output_schema)
    serializer = PyTorchOutputSerializer(schema=output_schema)
    import torch
    tensor = torch.randn(3)
    dummy_output_serialized = serializer.serialize(tensor)
    # assert len(dummy_output_serialized) == len(output_schema.outputs)
    # dummy_output_d = serializer.(dummy_output_serialized)
    # assert len(dummy_output_d.outputs) == len(dummy_output.outputs)


if __name__ == "__main__":
    test_serialize_schema()
    test_serialize_output_schema()
