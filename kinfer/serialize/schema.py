"""Defines utility functions for the schema."""

import numpy as np

from kinfer.protos.kinfer_pb2 import (
    AudioFrameValue,
    CameraFrameValue,
    IMUValue,
    Input,
    InputSchema,
    JointPositionsValue,
    JointPositionValue,
    JointTorquesValue,
    JointTorqueValue,
    JointVelocitiesValue,
    JointVelocityValue,
    Output,
    OutputSchema,
    TensorValue,
    TimestampValue,
    Value,
    ValueSchema,
)


def get_dummy_value(value_schema: ValueSchema) -> Value:
    value_type = value_schema.WhichOneof("value_type")

    match value_type:
        case "tensor":
            return Value(
                tensor=TensorValue(data=[0.0] * np.prod(value_schema.tensor.shape)),
            )
        case "joint_positions":
            return Value(
                joint_positions=JointPositionsValue(
                    values=[
                        JointPositionValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_positions.unit,
                        )
                        for joint_name in value_schema.joint_positions.joint_names
                    ]
                ),
            )
        case "joint_velocities":
            return Value(
                joint_velocities=JointVelocitiesValue(
                    values=[
                        JointVelocityValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_velocities.unit,
                        )
                        for joint_name in value_schema.joint_velocities.joint_names
                    ]
                ),
            )
        case "joint_torques":
            return Value(
                joint_torques=JointTorquesValue(
                    values=[
                        JointTorqueValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_torques.unit,
                        )
                        for joint_name in value_schema.joint_torques.joint_names
                    ]
                ),
            )
        case "camera_frame":
            return Value(
                camera_frame=CameraFrameValue(
                    data=b"\x00"
                    * (
                        value_schema.camera_frame.width
                        * value_schema.camera_frame.height
                        * value_schema.camera_frame.channels
                    )
                ),
            )
        case "audio_frame":
            return Value(
                audio_frame=AudioFrameValue(
                    data=b"\x00"
                    * (
                        value_schema.audio_frame.channels
                        * value_schema.audio_frame.sample_rate
                        * value_schema.audio_frame.bytes_per_sample
                    )
                ),
            )
        case "imu":
            return Value(
                imu=IMUValue(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        case "timestamp":
            return Value(
                timestamp=TimestampValue(seconds=1728000000, nanos=0),
            )
        case _:
            raise ValueError(f"Invalid value type: {value_type}")


def get_dummy_inputs(input_schema: InputSchema) -> Input:
    input_value = Input()
    for value_schema in input_schema.inputs:
        input_value.inputs.append(get_dummy_value(value_schema))
    return input_value


def get_dummy_outputs(output_schema: OutputSchema) -> Output:
    output_value = Output()
    for value_schema in output_schema.outputs:
        output_value.outputs.append(get_dummy_value(value_schema))
    return output_value
