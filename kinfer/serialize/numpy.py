"""Defines a serializer for Numpy arrays."""

from typing import cast

import numpy as np

from kinfer import proto as K
from kinfer.serialize.base import (
    AudioFrameSerializer,
    CameraFrameSerializer,
    ImuSerializer,
    JointCommandsSerializer,
    JointPositionsSerializer,
    JointTorquesSerializer,
    JointVelocitiesSerializer,
    MultiSerializer,
    Serializer,
    StateTensorSerializer,
    TimestampSerializer,
    VectorCommandSerializer,
)
from kinfer.serialize.utils import (
    as_float,
    check_names_match,
    convert_angular_position,
    convert_angular_velocity,
    convert_torque,
    dtype_num_bytes,
    dtype_range,
    numpy_dtype,
    parse_bytes,
)


class NumpyBaseSerializer:
    def __init__(self: "NumpyBaseSerializer", dtype: np.dtype | None = None) -> None:
        self.dtype = dtype or np.float32


class NumpyJointPositionsSerializer(NumpyBaseSerializer, JointPositionsSerializer[np.ndarray]):
    def serialize_joint_positions(
        self: "NumpyJointPositionsSerializer",
        schema: K.JointPositionsSchema,
        value: K.JointPositionsValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        array = np.array(
            [
                convert_angular_position(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ],
            dtype=self.dtype,
        )

        return array

    def deserialize_joint_positions(
        self: "NumpyJointPositionsSerializer",
        schema: K.JointPositionsSchema,
        value: np.ndarray,
    ) -> K.JointPositionsValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of array must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = cast(list[float], value.astype(float).tolist())
        return K.JointPositionsValue(
            values=[
                K.JointPositionValue(
                    joint_name=name,
                    value=float(value_list[i]),
                    unit=schema.unit,
                )
                for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyJointVelocitiesSerializer(NumpyBaseSerializer, JointVelocitiesSerializer[np.ndarray]):
    def serialize_joint_velocities(
        self: "NumpyJointVelocitiesSerializer",
        schema: K.JointVelocitiesSchema,
        value: K.JointVelocitiesValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        array = np.array(
            [
                convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ],
            dtype=self.dtype,
        )
        return array

    def deserialize_joint_velocities(
        self: "NumpyJointVelocitiesSerializer",
        schema: K.JointVelocitiesSchema,
        value: np.ndarray,
    ) -> K.JointVelocitiesValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of array must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = cast(list[float], value.astype(float).tolist())
        return K.JointVelocitiesValue(
            values=[
                K.JointVelocityValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyJointTorquesSerializer(NumpyBaseSerializer, JointTorquesSerializer[np.ndarray]):
    def serialize_joint_torques(
        self: "NumpyJointTorquesSerializer",
        schema: K.JointTorquesSchema,
        value: K.JointTorquesValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        array = np.array(
            [convert_torque(value_map[name].value, value_map[name].unit, schema.unit) for name in schema.joint_names],
            dtype=self.dtype,
        )
        return array

    def deserialize_joint_torques(
        self: "NumpyJointTorquesSerializer",
        schema: K.JointTorquesSchema,
        value: np.ndarray,
    ) -> K.JointTorquesValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of array must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = cast(list[float], value.astype(float).tolist())
        return K.JointTorquesValue(
            values=[
                K.JointTorqueValue(joint_name=name, value=float(value_list[i]), unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyJointCommandsSerializer(NumpyBaseSerializer, JointCommandsSerializer[np.ndarray]):
    def _convert_value_to_array(
        self: "NumpyJointCommandsSerializer",
        value: K.JointCommandValue,
        schema: K.JointCommandsSchema,
    ) -> np.ndarray:
        return np.array(
            [
                convert_torque(value.torque, value.torque_unit, schema.torque_unit),
                convert_angular_velocity(value.velocity, value.velocity_unit, schema.velocity_unit),
                convert_angular_position(value.position, value.position_unit, schema.position_unit),
                value.kp,
                value.kd,
            ],
            dtype=self.dtype,
        )

    def _convert_array_to_value(
        self: "NumpyJointCommandsSerializer",
        values: list[float],
        schema: K.JointCommandsSchema,
        name: str,
    ) -> K.JointCommandValue:
        if len(values) != 5:
            raise ValueError(f"Shape of array must match number of joint commands: {len(values)} != 5")
        return K.JointCommandValue(
            joint_name=name,
            torque=values[0],
            velocity=values[1],
            position=values[2],
            kp=values[3],
            kd=values[4],
            torque_unit=schema.torque_unit,
            velocity_unit=schema.velocity_unit,
            position_unit=schema.position_unit,
        )

    def serialize_joint_commands(
        self: "NumpyJointCommandsSerializer",
        schema: K.JointCommandsSchema,
        value: K.JointCommandsValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", list(value_map.keys()))
        array = np.stack(
            [self._convert_value_to_array(value_map[name], schema) for name in schema.joint_names],
            axis=0,
        )
        return array

    def deserialize_joint_commands(
        self: "NumpyJointCommandsSerializer",
        schema: K.JointCommandsSchema,
        value: np.ndarray,
    ) -> K.JointCommandsValue:
        if value.shape != (len(schema.joint_names), 5):
            raise ValueError(
                "Shape of array must match number of joint names and commands: "
                f"{value.shape} != ({len(schema.joint_names)}, 5)"
            )
        value_list = cast(list[list[float]], value.astype(float).tolist())
        return K.JointCommandsValue(
            values=[
                self._convert_array_to_value(value_list[i], schema, name) for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyCameraFrameSerializer(NumpyBaseSerializer, CameraFrameSerializer[np.ndarray]):
    def serialize_camera_frame(
        self: "NumpyCameraFrameSerializer",
        schema: K.CameraFrameSchema,
        value: K.CameraFrameValue,
    ) -> np.ndarray:
        np_arr = parse_bytes(value.data, K.DType.UINT8)
        array = np_arr.astype(self.dtype) / 255.0
        if array.size != schema.channels * schema.height * schema.width:
            raise ValueError(
                "Length of data must match number of channels, height, and width: "
                f"{array.size} != {schema.channels} * {schema.height} * {schema.width}"
            )
        array = array.reshape(schema.channels, schema.height, schema.width)
        return array

    def deserialize_camera_frame(
        self: "NumpyCameraFrameSerializer",
        schema: K.CameraFrameSchema,
        value: np.ndarray,
    ) -> K.CameraFrameValue:
        np_arr = (value * 255.0).flatten().astype(np.uint8)
        return K.CameraFrameValue(data=np_arr.tobytes())


class NumpyAudioFrameSerializer(NumpyBaseSerializer, AudioFrameSerializer[np.ndarray]):
    def serialize_audio_frame(
        self: "NumpyAudioFrameSerializer",
        schema: K.AudioFrameSchema,
        value: K.AudioFrameValue,
    ) -> np.ndarray:
        value_bytes = value.data
        if len(value_bytes) != schema.channels * schema.sample_rate * dtype_num_bytes(schema.dtype):
            raise ValueError(
                "Length of data must match number of channels, sample rate, and dtype: "
                f"{len(value_bytes)} != {schema.channels} * {schema.sample_rate} * {dtype_num_bytes(schema.dtype)}"
            )
        _, max_value = dtype_range(schema.dtype)
        np_arr = parse_bytes(value_bytes, schema.dtype)
        array = np_arr.astype(self.dtype)
        array = array.reshape(schema.channels, -1)
        array = array / max_value
        return array

    def deserialize_audio_frame(
        self: "NumpyAudioFrameSerializer",
        schema: K.AudioFrameSchema,
        value: np.ndarray,
    ) -> K.AudioFrameValue:
        _, max_value = dtype_range(schema.dtype)
        np_arr = (value * max_value).flatten().astype(numpy_dtype(schema.dtype))
        return K.AudioFrameValue(data=np_arr.tobytes())


class NumpyImuSerializer(NumpyBaseSerializer, ImuSerializer[np.ndarray]):
    def serialize_imu(
        self: "NumpyImuSerializer",
        schema: K.ImuSchema,
        value: K.ImuValue,
    ) -> np.ndarray:
        vectors = []
        if schema.use_accelerometer:
            vectors.append(
                np.array(
                    [value.linear_acceleration.x, value.linear_acceleration.y, value.linear_acceleration.z],
                    dtype=self.dtype,
                )
            )
        if schema.use_gyroscope:
            vectors.append(
                np.array(
                    [value.angular_velocity.x, value.angular_velocity.y, value.angular_velocity.z],
                    dtype=self.dtype,
                )
            )
        if schema.use_magnetometer:
            vectors.append(
                np.array(
                    [value.magnetic_field.x, value.magnetic_field.y, value.magnetic_field.z],
                    dtype=self.dtype,
                )
            )
        if not vectors:
            raise ValueError("IMU has nothing to serialize")
        return np.stack(vectors, axis=0)

    def deserialize_imu(
        self: "NumpyImuSerializer",
        schema: K.ImuSchema,
        value: np.ndarray,
    ) -> K.ImuValue:
        num_vectors = sum([schema.use_accelerometer, schema.use_gyroscope, schema.use_magnetometer])
        if value.shape != (num_vectors, 3):
            raise ValueError(
                f"Shape of array must match number of vectors and components: {value.shape} != ({num_vectors}, 3)"
            )
        vectors = cast(list[list[float]], value.astype(float).tolist())
        imu_value = K.ImuValue()
        if schema.use_accelerometer:
            x, y, z = vectors.pop(0)
            imu_value.linear_acceleration.x = as_float(x)
            imu_value.linear_acceleration.y = as_float(y)
            imu_value.linear_acceleration.z = as_float(z)
        if schema.use_gyroscope:
            x, y, z = vectors.pop(0)
            imu_value.angular_velocity.x = as_float(x)
            imu_value.angular_velocity.y = as_float(y)
            imu_value.angular_velocity.z = as_float(z)
        if schema.use_magnetometer:
            x, y, z = vectors.pop(0)
            imu_value.magnetic_field.x = as_float(x)
            imu_value.magnetic_field.y = as_float(y)
            imu_value.magnetic_field.z = as_float(z)
        return imu_value


class NumpyTimestampSerializer(NumpyBaseSerializer, TimestampSerializer[np.ndarray]):
    def serialize_timestamp(
        self: "NumpyTimestampSerializer",
        schema: K.TimestampSchema,
        value: K.TimestampValue,
    ) -> np.ndarray:
        elapsed_seconds = value.seconds - schema.start_seconds
        elapsed_nanos = value.nanos - schema.start_nanos
        if elapsed_nanos < 0:
            elapsed_seconds -= 1
            elapsed_nanos += 1_000_000_000
        total_elapsed_seconds = elapsed_seconds + elapsed_nanos / 1_000_000_000
        return np.array([total_elapsed_seconds], dtype=self.dtype)

    def deserialize_timestamp(
        self: "NumpyTimestampSerializer",
        schema: K.TimestampSchema,
        value: np.ndarray,
    ) -> K.TimestampValue:
        total_elapsed_seconds = float(value.item())
        elapsed_seconds = int(total_elapsed_seconds)
        elapsed_nanos = int((total_elapsed_seconds - elapsed_seconds) * 1_000_000_000)
        return K.TimestampValue(seconds=elapsed_seconds, nanos=elapsed_nanos)


class NumpyVectorCommandSerializer(NumpyBaseSerializer, VectorCommandSerializer[np.ndarray]):
    def serialize_vector_command(
        self: "NumpyVectorCommandSerializer",
        schema: K.VectorCommandSchema,
        value: K.VectorCommandValue,
    ) -> np.ndarray:
        return np.array(value.values, dtype=self.dtype)

    def deserialize_vector_command(
        self: "NumpyVectorCommandSerializer",
        schema: K.VectorCommandSchema,
        value: np.ndarray,
    ) -> K.VectorCommandValue:
        if value.shape != (schema.dimensions,):
            raise ValueError(f"Shape of array must match number of dimensions: {value.shape} != {schema.dimensions}")
        values = cast(list[float], value.astype(float).tolist())
        return K.VectorCommandValue(values=values)


class NumpyStateTensorSerializer(NumpyBaseSerializer, StateTensorSerializer[np.ndarray]):
    def serialize_state_tensor(
        self: "NumpyStateTensorSerializer",
        schema: K.StateTensorSchema,
        value: K.StateTensorValue,
    ) -> np.ndarray:
        value_bytes = value.data
        if len(value_bytes) != np.prod(schema.shape) * dtype_num_bytes(schema.dtype):
            raise ValueError(
                "Length of data must match number of elements: "
                f"{len(value_bytes)} != {np.prod(schema.shape)} * {dtype_num_bytes(schema.dtype)}"
            )
        np_arr = parse_bytes(value_bytes, schema.dtype)
        array = np.ascontiguousarray(np_arr.astype(numpy_dtype(schema.dtype)))
        array = array.reshape(tuple(schema.shape))
        return array

    def deserialize_state_tensor(
        self: "NumpyStateTensorSerializer",
        schema: K.StateTensorSchema,
        value: np.ndarray,
    ) -> K.StateTensorValue:
        contiguous_value = np.ascontiguousarray(value)
        return K.StateTensorValue(data=contiguous_value.flatten().tobytes())


class NumpySerializer(
    NumpyJointPositionsSerializer,
    NumpyJointVelocitiesSerializer,
    NumpyJointTorquesSerializer,
    NumpyJointCommandsSerializer,
    NumpyCameraFrameSerializer,
    NumpyAudioFrameSerializer,
    NumpyImuSerializer,
    NumpyTimestampSerializer,
    NumpyVectorCommandSerializer,
    NumpyStateTensorSerializer,
    Serializer[np.ndarray],
):
    def __init__(
        self: "NumpySerializer",
        schema: K.ValueSchema,
        *,
        dtype: np.dtype | None = None,
    ) -> None:
        NumpyBaseSerializer.__init__(self, dtype=dtype)
        Serializer.__init__(self, schema=schema)


class NumpyMultiSerializer(MultiSerializer[np.ndarray]):
    def __init__(self: "NumpyMultiSerializer", schema: K.IOSchema) -> None:
        super().__init__([NumpySerializer(schema=s) for s in schema.values])
