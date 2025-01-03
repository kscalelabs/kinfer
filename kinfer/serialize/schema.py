"""Defines utility functions for the schema."""

import numpy as np

from kinfer import proto as K
from kinfer.serialize.utils import dtype_num_bytes


def get_dummy_value(value_schema: K.ValueSchema) -> K.Value:
    value_type = value_schema.WhichOneof("value_type")

    match value_type:
        case "joint_positions":
            return K.Value(
                joint_positions=K.JointPositionsValue(
                    values=[
                        K.JointPositionValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_positions.unit,
                        )
                        for joint_name in value_schema.joint_positions.joint_names
                    ]
                ),
            )
        case "joint_velocities":
            return K.Value(
                joint_velocities=K.JointVelocitiesValue(
                    values=[
                        K.JointVelocityValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_velocities.unit,
                        )
                        for joint_name in value_schema.joint_velocities.joint_names
                    ]
                ),
            )
        case "joint_torques":
            return K.Value(
                joint_torques=K.JointTorquesValue(
                    values=[
                        K.JointTorqueValue(
                            joint_name=joint_name,
                            value=0.0,
                            unit=value_schema.joint_torques.unit,
                        )
                        for joint_name in value_schema.joint_torques.joint_names
                    ]
                ),
            )
        case "joint_commands":
            return K.Value(
                joint_commands=K.JointCommandsValue(
                    values=[
                        K.JointCommandValue(
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
            return K.Value(
                camera_frame=K.CameraFrameValue(
                    data=b"\x00"
                    * (
                        value_schema.camera_frame.width
                        * value_schema.camera_frame.height
                        * value_schema.camera_frame.channels
                    )
                ),
            )
        case "audio_frame":
            return K.Value(
                audio_frame=K.AudioFrameValue(
                    data=b"\x00"
                    * (
                        value_schema.audio_frame.channels
                        * value_schema.audio_frame.sample_rate
                        * dtype_num_bytes(value_schema.audio_frame.dtype)
                    )
                ),
            )
        case "imu":
            return K.Value(
                imu=K.ImuValue(
                    linear_acceleration=K.ImuAccelerometerValue(x=0.0, y=0.0, z=0.0),
                    angular_velocity=K.ImuGyroscopeValue(x=0.0, y=0.0, z=0.0),
                    magnetic_field=K.ImuMagnetometerValue(x=0.0, y=0.0, z=0.0),
                ),
            )
        case "timestamp":
            return K.Value(
                timestamp=K.TimestampValue(seconds=1728000000, nanos=0),
            )
        case "vector_command":
            return K.Value(
                vector_command=K.VectorCommandValue(values=[0.0] * value_schema.vector_command.dimensions),
            )
        case "state_tensor":
            return K.Value(
                state_tensor=K.StateTensorValue(
                    data=b"\x00"
                    * np.prod(value_schema.state_tensor.shape)
                    * dtype_num_bytes(value_schema.state_tensor.dtype)
                ),
            )
        case _:
            raise ValueError(f"Invalid value type: {value_type}")


def get_dummy_io(schema: K.IOSchema) -> K.IO:
    io_value = K.IO()
    for value_schema in schema.values:
        io_value.values.append(get_dummy_value(value_schema))
    return io_value
