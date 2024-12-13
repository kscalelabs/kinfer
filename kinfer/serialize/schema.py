"""Defines utility functions for the schema."""

from typing import Literal, overload

from google.protobuf.json_format import ParseDict

from kinfer.protos.kinfer_pb2 import (
    AudioFrameValue,
    CameraFrameValue,
    IMUAccelerometerValue,
    IMUGyroscopeValue,
    IMUMagnetometerValue,
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
    TimestampValue,
    Value,
    ValueSchema,
    VectorCommandValue,
)
from kinfer.serialize.utils import dtype_num_bytes

ValueSchemaType = Literal[
    "joint_positions",
    "joint_velocities",
    "joint_torques",
    "joint_commands",
    "camera_frame",
    "audio_frame",
    "imu",
    "timestamp",
    "vector_command",
    "state_tensor",
]


def get_dummy_value(value_schema: ValueSchema) -> Value:
    value_type = value_schema.WhichOneof("value_type")

    match value_type:
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
                        * dtype_num_bytes(value_schema.audio_frame.dtype)
                    )
                ),
            )
        case "imu":
            return Value(
                imu=IMUValue(
                    linear_acceleration=IMUAccelerometerValue(x=0.0, y=0.0, z=0.0),
                    angular_velocity=IMUGyroscopeValue(x=0.0, y=0.0, z=0.0),
                    magnetic_field=IMUMagnetometerValue(x=0.0, y=0.0, z=0.0),
                ),
            )
        case "timestamp":
            return Value(
                timestamp=TimestampValue(seconds=1728000000, nanos=0),
            )
        case "vector_command":
            return Value(
                vector_command=VectorCommandValue(values=[0.0, 0.0, 0.0]),
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


@overload
def parse_schema(values: dict[ValueSchemaType, ValueSchema], mode: Literal["input"]) -> InputSchema: ...


@overload
def parse_schema(values: dict[ValueSchemaType, ValueSchema], mode: Literal["output"]) -> OutputSchema: ...


def parse_schema(
    values: dict[ValueSchemaType, ValueSchema],
    mode: Literal["input", "output"],
) -> InputSchema | OutputSchema:
    schema = InputSchema() if mode == "input" else OutputSchema()
    return ParseDict(values, schema)
