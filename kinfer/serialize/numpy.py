"""Defines a serializer for Numpy arrays."""

import numpy as np

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    AudioFrameValue,
    CameraFrameSchema,
    CameraFrameValue,
    DType,
    IMUSchema,
    IMUValue,
    InputSchema,
    JointCommandsSchema,
    JointCommandsValue,
    JointCommandValue,
    JointPositionsSchema,
    JointPositionsValue,
    JointPositionValue,
    JointTorquesSchema,
    JointTorquesValue,
    JointTorqueValue,
    JointVelocitiesSchema,
    JointVelocitiesValue,
    JointVelocityValue,
    OutputSchema,
    StateTensorSchema,
    StateTensorValue,
    TimestampSchema,
    TimestampValue,
    ValueSchema,
    VectorCommandSchema,
    VectorCommandValue,
)
from kinfer.serialize.base import (
    AudioFrameSerializer,
    CameraFrameSerializer,
    IMUSerializer,
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
    def __init__(self, dtype: np.dtype | None = None) -> None:
        self.dtype = dtype


class NumpyJointPositionsSerializer(NumpyBaseSerializer, JointPositionsSerializer[np.ndarray]):
    def serialize_joint_positions(
        self,
        schema: JointPositionsSchema,
        value: JointPositionsValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        array = np.array(
            [
                convert_angular_position(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ],
            dtype=self.dtype,
        )
        return array

    def deserialize_joint_positions(
        self,
        schema: JointPositionsSchema,
        value: np.ndarray,
    ) -> JointPositionsValue:
        if value.shape != (len(schema.joint_names),):
            raise ValueError(
                f"Shape of array must match number of joint names: {value.shape} != {len(schema.joint_names)}"
            )
        value_list = value.flatten().tolist()
        return JointPositionsValue(
            values=[
                JointPositionValue(
                    joint_name=name,
                    value=value_list[i],
                    unit=schema.unit,
                )
                for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyJointVelocitiesSerializer(NumpyBaseSerializer, JointVelocitiesSerializer[np.ndarray]):
    def serialize_joint_velocities(
        self,
        schema: JointVelocitiesSchema,
        value: JointVelocitiesValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        array = np.array(
            [
                convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ],
            dtype=self.dtype,
        )
        return array

    def deserialize_joint_velocities(
        self,
        schema: JointVelocitiesSchema,
        value: np.ndarray,
    ) -> JointVelocitiesValue:
        value_list = value.flatten().tolist()
        return JointVelocitiesValue(
            values=[
                JointVelocityValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyJointTorquesSerializer(NumpyBaseSerializer, JointTorquesSerializer[np.ndarray]):
    def serialize_joint_torques(
        self,
        schema: JointTorquesSchema,
        value: JointTorquesValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        array = np.array(
            [convert_torque(value_map[name].value, value_map[name].unit, schema.unit) for name in schema.joint_names],
            dtype=self.dtype,
        )
        return array

    def deserialize_joint_torques(
        self,
        schema: JointTorquesSchema,
        value: np.ndarray,
    ) -> JointTorquesValue:
        value_list = value.flatten().tolist()
        return JointTorquesValue(
            values=[
                JointTorqueValue(joint_name=name, value=value_list[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyJointCommandsSerializer(NumpyBaseSerializer, JointCommandsSerializer[np.ndarray]):
    def _convert_value_to_array(
        self,
        value: JointCommandValue,
        schema: JointCommandsSchema,
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
        self,
        values: list[float],
        schema: JointCommandsSchema,
        name: str,
    ) -> JointCommandValue:
        if len(values) != 5:
            raise ValueError(f"Shape of array must match number of joint commands: {len(values)} != 5")
        return JointCommandValue(
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
        self,
        schema: JointCommandsSchema,
        value: JointCommandsValue,
    ) -> np.ndarray:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        array = np.stack(
            [self._convert_value_to_array(value_map[name], schema) for name in schema.joint_names],
            axis=0,
        )
        return array

    def deserialize_joint_commands(self, schema: JointCommandsSchema, value: np.ndarray) -> JointCommandsValue:
        if value.shape != (len(schema.joint_names), 5):
            raise ValueError(
                "Shape of array must match number of joint names and commands: "
                f"{value.shape} != ({len(schema.joint_names)}, 5)"
            )
        value_list = value.tolist()
        return JointCommandsValue(
            values=[
                self._convert_array_to_value(value_list[i], schema, name) for i, name in enumerate(schema.joint_names)
            ]
        )


class NumpyCameraFrameSerializer(NumpyBaseSerializer, CameraFrameSerializer[np.ndarray]):
    def serialize_camera_frame(self, schema: CameraFrameSchema, value: CameraFrameValue) -> np.ndarray:
        np_arr = parse_bytes(value.data, DType.UINT8)
        array = np_arr.astype(self.dtype) / 255.0
        if array.size != schema.channels * schema.height * schema.width:
            raise ValueError(
                "Length of data must match number of channels, height, and width: "
                f"{array.size} != {schema.channels} * {schema.height} * {schema.width}"
            )
        array = array.reshape(schema.channels, schema.height, schema.width)
        return array

    def deserialize_camera_frame(self, schema: CameraFrameSchema, value: np.ndarray) -> CameraFrameValue:
        np_arr = (value * 255.0).flatten().astype(np.uint8)
        return CameraFrameValue(data=np_arr.tobytes())


class NumpyAudioFrameSerializer(NumpyBaseSerializer, AudioFrameSerializer[np.ndarray]):
    def serialize_audio_frame(self, schema: AudioFrameSchema, value: AudioFrameValue) -> np.ndarray:
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

    def deserialize_audio_frame(self, schema: AudioFrameSchema, value: np.ndarray) -> AudioFrameValue:
        _, max_value = dtype_range(schema.dtype)
        np_arr = (value * max_value).flatten().astype(numpy_dtype(schema.dtype))
        return AudioFrameValue(data=np_arr.tobytes())


class NumpyIMUSerializer(NumpyBaseSerializer, IMUSerializer[np.ndarray]):
    def serialize_imu(self, schema: IMUSchema, value: IMUValue) -> np.ndarray:
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

    def deserialize_imu(self, schema: IMUSchema, value: np.ndarray) -> IMUValue:
        vectors = value.tolist()
        imu_value = IMUValue()
        if schema.use_accelerometer:
            (x, y, z), vectors = vectors[0], vectors[1:]
            imu_value.linear_acceleration.x = x
            imu_value.linear_acceleration.y = y
            imu_value.linear_acceleration.z = z
        if schema.use_gyroscope:
            (x, y, z), vectors = vectors[0], vectors[1:]
            imu_value.angular_velocity.x = x
            imu_value.angular_velocity.y = y
            imu_value.angular_velocity.z = z
        if schema.use_magnetometer:
            (x, y, z), vectors = vectors[0], vectors[1:]
            imu_value.magnetic_field.x = x
            imu_value.magnetic_field.y = y
            imu_value.magnetic_field.z = z
        return imu_value


class NumpyTimestampSerializer(NumpyBaseSerializer, TimestampSerializer[np.ndarray]):
    def serialize_timestamp(self, schema: TimestampSchema, value: TimestampValue) -> np.ndarray:
        elapsed_seconds = value.seconds - schema.start_seconds
        elapsed_nanos = value.nanos - schema.start_nanos
        if elapsed_nanos < 0:
            elapsed_seconds -= 1
            elapsed_nanos += 1_000_000_000
        total_elapsed_seconds = elapsed_seconds + elapsed_nanos / 1_000_000_000
        return np.array([total_elapsed_seconds], dtype=self.dtype)

    def deserialize_timestamp(self, schema: TimestampSchema, value: np.ndarray) -> TimestampValue:
        total_elapsed_seconds = value.item()
        elapsed_seconds = int(total_elapsed_seconds)
        elapsed_nanos = int((total_elapsed_seconds - elapsed_seconds) * 1_000_000_000)
        return TimestampValue(seconds=elapsed_seconds, nanos=elapsed_nanos)


class NumpyVectorCommandSerializer(NumpyBaseSerializer, VectorCommandSerializer[np.ndarray]):
    def serialize_vector_command(self, schema: VectorCommandSchema, value: VectorCommandValue) -> np.ndarray:
        return np.array(value.values, dtype=self.dtype)

    def deserialize_vector_command(self, schema: VectorCommandSchema, value: np.ndarray) -> VectorCommandValue:
        return VectorCommandValue(values=value.tolist())


class NumpyStateTensorSerializer(NumpyBaseSerializer, StateTensorSerializer[np.ndarray]):
    def serialize_state_tensor(self, schema: StateTensorSchema, value: StateTensorValue) -> np.ndarray:
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

    def deserialize_state_tensor(self, schema: StateTensorSchema, value: np.ndarray) -> StateTensorValue:
        contiguous_value = np.ascontiguousarray(value)
        return StateTensorValue(data=contiguous_value.flatten().tobytes())


class NumpySerializer(
    NumpyJointPositionsSerializer,
    NumpyJointVelocitiesSerializer,
    NumpyJointTorquesSerializer,
    NumpyJointCommandsSerializer,
    NumpyCameraFrameSerializer,
    NumpyAudioFrameSerializer,
    NumpyIMUSerializer,
    NumpyTimestampSerializer,
    NumpyVectorCommandSerializer,
    NumpyStateTensorSerializer,
    Serializer[np.ndarray],
):
    def __init__(
        self,
        schema: ValueSchema,
        *,
        dtype: np.dtype | None = None,
    ) -> None:
        NumpyBaseSerializer.__init__(self, dtype=dtype)
        Serializer.__init__(self, schema=schema)


class NumpyMultiSerializer(MultiSerializer[np.ndarray]):
    def __init__(self, schema: InputSchema | OutputSchema) -> None:
        super().__init__([NumpySerializer(schema=s) for s in schema.inputs])
