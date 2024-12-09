"""Defines functions for serializing and deserializing signatures."""

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from kinfer.protos.kinfer_pb2 import (
    AudioFrameSchema,
    AudioFrameValue,
    CameraFrameSchema,
    CameraFrameValue,
    IMUSchema,
    IMUValue,
    Input,
    JointPositionsSchema,
    JointPositionsValue,
    JointTorquesSchema,
    JointTorquesValue,
    JointVelocitiesSchema,
    JointVelocitiesValue,
    TensorSchema,
    TensorValue,
    TimestampSchema,
    TimestampValue,
    Value,
    ValueSchema,
)

T = TypeVar("T")


class TensorSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_tensor(
        self,
        schema: TensorSchema,
        value: TensorValue,
    ) -> T:
        """Serialize a tensor value.

        Args:
            schema: The schema of the tensor.
            value: The tensor to serialize.

        Returns:
            The serialized tensor.
        """

    @abstractmethod
    def deserialize_tensor(
        self,
        schema: TensorSchema,
        value: T,
    ) -> TensorValue:
        """Deserialize a tensor value.

        Args:
            schema: The schema of the tensor.
            value: The serialized tensor.

        Returns:
            The deserialized tensor.
        """


class JointPositionsSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_positions(
        self,
        schema: JointPositionsSchema,
        value: JointPositionsValue,
    ) -> T:
        """Serialize a joint positions value.

        Args:
            schema: The schema of the joint positions.
            value: The joint positions to serialize.

        Returns:
            The serialized joint positions.
        """

    @abstractmethod
    def deserialize_joint_positions(
        self,
        schema: JointPositionsSchema,
        value: T,
    ) -> JointPositionsValue:
        """Deserialize a joint positions value.

        Args:
            schema: The schema of the joint positions.
            value: The serialized joint positions.
            radians: Whether the serialized joint positions are radians.

        Returns:
            The deserialized joint positions.
        """


class JointVelocitiesSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_velocities(
        self,
        schema: JointVelocitiesSchema,
        value: JointVelocitiesValue,
    ) -> T:
        """Serialize a joint velocities value.

        Args:
            schema: The schema of the joint velocities.
            value: The joint velocities to serialize.

        Returns:
            The serialized joint velocities.
        """

    @abstractmethod
    def deserialize_joint_velocities(
        self,
        schema: JointVelocitiesSchema,
        value: T,
    ) -> JointVelocitiesValue:
        """Deserialize a joint velocities value.

        Args:
            schema: The schema of the joint velocities.
            value: The serialized joint velocities.

        Returns:
            The deserialized joint velocities.
        """


class JointTorquesSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_torques(
        self,
        schema: JointTorquesSchema,
        value: JointTorquesValue,
    ) -> T:
        """Serialize a joint torques value.

        Args:
            schema: The schema of the joint torques.
            value: The joint torques to serialize.

        Returns:
            The serialized joint torques.
        """

    @abstractmethod
    def deserialize_joint_torques(
        self,
        schema: JointTorquesSchema,
        value: T,
    ) -> JointTorquesValue:
        """Deserialize a joint torques value.

        Args:
            schema: The schema of the joint torques.
            value: The serialized joint torques.

        Returns:
            The deserialized joint torques.
        """


class CameraFrameSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_camera_frame(
        self,
        schema: CameraFrameSchema,
        value: CameraFrameValue,
    ) -> T:
        """Serialize a camera frame value.

        Args:
            schema: The schema of the camera frame.
            value: The frame of camera to serialize.

        Returns:
            The serialized camera frame.
        """

    @abstractmethod
    def deserialize_camera_frame(
        self,
        schema: CameraFrameSchema,
        value: T,
    ) -> CameraFrameValue:
        """Deserialize a camera frame value.

        Args:
            schema: The schema of the camera frame.
            value: The serialized camera frame.

        Returns:
            The deserialized camera frame.
        """


class AudioFrameSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_audio_frame(
        self,
        schema: AudioFrameSchema,
        value: AudioFrameValue,
    ) -> T:
        """Serialize an audio frame value.

        Args:
            schema: The schema of the audio frame.
            value: The frame of audio to serialize.

        Returns:
            The serialized audio frame.
        """

    @abstractmethod
    def deserialize_audio_frame(
        self,
        schema: AudioFrameSchema,
        value: T,
    ) -> AudioFrameValue:
        """Deserialize an audio frame value.

        Args:
            schema: The schema of the audio frame.
            value: The serialized audio frame.

        Returns:
            The deserialized audio frame.
        """


class IMUSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_imu(
        self,
        schema: IMUSchema,
        value: IMUValue,
    ) -> T:
        """Serialize an IMU value.

        Args:
            schema: The schema of the IMU.
            value: The IMU to serialize.

        Returns:
            The serialized IMU.
        """

    @abstractmethod
    def deserialize_imu(
        self,
        schema: IMUSchema,
        value: T,
    ) -> IMUValue:
        """Deserialize an IMU value.

        Args:
            schema: The schema of the IMU.
            value: The serialized IMU.

        Returns:
            The deserialized IMU.
        """


class TimestampSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_timestamp(
        self,
        schema: TimestampSchema,
        value: TimestampValue,
    ) -> T:
        """Serialize a timestamp value.

        Args:
            schema: The schema of the timestamp.
            value: The timestamp to serialize.

        Returns:
            The serialized timestamp.
        """

    @abstractmethod
    def deserialize_timestamp(
        self,
        schema: TimestampSchema,
        value: T,
    ) -> TimestampValue:
        """Deserialize a timestamp value.

        Args:
            schema: The schema of the timestamp.
            value: The serialized timestamp.

        Returns:
            The deserialized timestamp.
        """


class Serializer(
    TensorSerializer[T],
    JointPositionsSerializer[T],
    JointVelocitiesSerializer[T],
    JointTorquesSerializer[T],
    CameraFrameSerializer[T],
    AudioFrameSerializer[T],
    IMUSerializer[T],
    TimestampSerializer[T],
    Generic[T],
):
    def __init__(self, schema: ValueSchema) -> None:
        self.schema = schema

    def serialize(self, value: Value) -> T:
        value_type = value.WhichOneof("value")

        match value_type:
            case "tensor":
                return self.serialize_tensor(
                    schema=self.schema.tensor,
                    value=value.tensor,
                )
            case "joint_positions":
                return self.serialize_joint_positions(
                    schema=self.schema.joint_positions,
                    value=value.joint_positions,
                )
            case "joint_velocities":
                return self.serialize_joint_velocities(
                    schema=self.schema.joint_velocities,
                    value=value.joint_velocities,
                )
            case "joint_torques":
                return self.serialize_joint_torques(
                    schema=self.schema.joint_torques,
                    value=value.joint_torques,
                )
            case "camera_frame":
                return self.serialize_camera_frame(
                    schema=self.schema.camera_frame,
                    value=value.camera_frame,
                )
            case "audio_frame":
                return self.serialize_audio_frame(
                    schema=self.schema.audio_frame,
                    value=value.audio_frame,
                )
            case "imu":
                return self.serialize_imu(
                    schema=self.schema.imu,
                    value=value.imu,
                )
            case "timestamp":
                return self.serialize_timestamp(
                    schema=self.schema.timestamp,
                    value=value.timestamp,
                )
            case _:
                raise ValueError(f"Unsupported value type: {value_type}")

    def deserialize(self, value: T) -> Value:
        value_type = self.schema.WhichOneof("value_type")

        match value_type:
            case "tensor":
                return Value(
                    tensor=self.deserialize_tensor(
                        schema=self.schema.tensor,
                        value=value,
                    ),
                )
            case "joint_positions":
                return Value(
                    joint_positions=self.deserialize_joint_positions(
                        schema=self.schema.joint_positions,
                        value=value,
                    ),
                )
            case "joint_velocities":
                return Value(
                    joint_velocities=self.deserialize_joint_velocities(
                        schema=self.schema.joint_velocities,
                        value=value,
                    ),
                )
            case "joint_torques":
                return Value(
                    joint_torques=self.deserialize_joint_torques(
                        schema=self.schema.joint_torques,
                        value=value,
                    ),
                )
            case "camera_frame":
                return Value(
                    camera_frame=self.deserialize_camera_frame(
                        schema=self.schema.camera_frame,
                        value=value,
                    ),
                )
            case "audio_frame":
                return Value(
                    audio_frame=self.deserialize_audio_frame(
                        schema=self.schema.audio_frame,
                        value=value,
                    ),
                )
            case "imu":
                return Value(
                    imu=self.deserialize_imu(
                        schema=self.schema.imu,
                        value=value,
                    ),
                )
            case "timestamp":
                return Value(
                    timestamp=self.deserialize_timestamp(
                        schema=self.schema.timestamp,
                        value=value,
                    ),
                )
            case _:
                raise ValueError(f"Unsupported value type: {value_type}")


class MultiSerializer(Generic[T]):
    def __init__(self, serializers: Sequence[Serializer[T]]) -> None:
        self.serializers = list(serializers)

    def serialize(self, input: Input) -> dict[str, T]:
        return {s.schema.value_name: s.serialize(i) for s, i in zip(self.serializers, input.inputs)}

    def deserialize(self, input: dict[str, T]) -> Input:
        return Input(inputs=[s.deserialize(i) for s, i in zip(self.serializers, input.items())])
