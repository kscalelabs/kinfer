"""Defines utility functions for the schema."""

import numpy as np

from kinfer import proto as P
from kinfer.serialize.utils import dtype_num_bytes


def get_dummy_value(value_schema: P.ValueSchema) -> P.Value:
    value_type = value_schema.WhichOneof("value_type")

    match value_type:
        case "joint_positions":
            return P.Value(
                joint_positions=P.JointPositionsValue(
                    values=[
                        P.JointPositionValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_positions.unit,
                        )
                        for joint_name in value_schema.joint_positions.joint_names
                    ]
                ),
            )
        case "joint_velocities":
            return P.Value(
                joint_velocities=P.JointVelocitiesValue(
                    values=[
                        P.JointVelocityValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_velocities.unit,
                        )
                        for joint_name in value_schema.joint_velocities.joint_names
                    ]
                ),
            )
        case "joint_torques":
            return P.Value(
                joint_torques=P.JointTorquesValue(
                    values=[
                        P.JointTorqueValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_torques.unit,
                        )
                        for joint_name in value_schema.joint_torques.joint_names
                    ]
                ),
            )
        case "joint_commands":
            return P.Value(
                joint_commands=P.JointCommandsValue(
                    values=[
                        P.JointCommandValue(
                            joint_name=joint_name,
                            torque=0.0,
                            velocity=0.0,
                            position=0.0,
                            kp=0.0,
                            kd=0.0,
                            torque_unit=value_schema.joint_commands.torque_unit,
                            velocity_unit=value_schema.joint_commands.velocity_unit,
                            position_unit=value_schema.joint_commands.position_unit,
                        )
                        for joint_name in value_schema.joint_commands.joint_names
                    ]
                ),
            )
        case "camera_frame":
            return P.Value(
                camera_frame=P.CameraFrameValue(
                    data=b"\x00"
                    * (
                        value_schema.camera_frame.width
                        * value_schema.camera_frame.height
                        * value_schema.camera_frame.channels
                    )
                ),
            )
        case "audio_frame":
            return P.Value(
                audio_frame=P.AudioFrameValue(
                    data=b"\x00"
                    * (
                        value_schema.audio_frame.channels
                        * value_schema.audio_frame.sample_rate
                        * dtype_num_bytes(value_schema.audio_frame.dtype)
                    )
                ),
            )
        case "imu":
            return P.Value(
                imu=P.ImuValue(
                    linear_acceleration=P.ImuAccelerometerValue(x=0.0, y=0.0, z=0.0),
                    angular_velocity=P.ImuGyroscopeValue(x=0.0, y=0.0, z=0.0),
                    magnetic_field=P.ImuMagnetometerValue(x=0.0, y=0.0, z=0.0),
                ),
            )
        case "timestamp":
            return P.Value(
                timestamp=P.TimestampValue(seconds=1728000000, nanos=0),
            )
        case "vector_command":
            return P.Value(
                vector_command=P.VectorCommandValue(values=[0.0] * value_schema.vector_command.dimensions),
            )
        case "state_tensor":
            return P.Value(
                state_tensor=P.StateTensorValue(
                    data=b"\x00"
                    * np.prod(value_schema.state_tensor.shape)
                    * dtype_num_bytes(value_schema.state_tensor.dtype)
                ),
            )
        case _:
            raise ValueError(f"Invalid value type: {value_type}")


def get_dummy_io(schema: P.IOSchema) -> P.IO:
    io_value = P.IO()
    for value_schema in schema.values:
        io_value.values.append(get_dummy_value(value_schema))
    return io_value
