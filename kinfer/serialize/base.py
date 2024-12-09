"""Defines functions for serializing and deserializing signatures."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from kinfer.protos.kinfer_pb2 import (
    AudioFrameValue,
    CameraFrameValue,
    JointPositionsValue,
    JointTorquesValue,
    TensorValue,
    TimestampValue,
)

T = TypeVar("T")


class TimestampSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_timestamp(self, timestamp: TimestampValue) -> T:
        """Serialize a timestamp value.

        Args:
            timestamp: The timestamp to serialize.

        Returns:
            The serialized timestamp.
        """

    @abstractmethod
    def deserialize_timestamp(self, value: T) -> TimestampValue:
        """Deserialize a timestamp value.

        Args:
            value: The serialized timestamp.

        Returns:
            The deserialized timestamp.
        """


class TensorSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_tensor(self, tensor: TensorValue) -> T:
        """Serialize a tensor value."""

    @abstractmethod
    def deserialize_tensor(self, value: T) -> TensorValue:
        """Deserialize a tensor value."""


class JointPositionsSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_positions(
        self,
        joint_positions: JointPositionsValue,
        *,
        radians: bool = False,
    ) -> tuple[T, list[str]]:
        """Serialize a joint positions value.

        Args:
            joint_positions: The joint positions to serialize.
            radians: Whether to serialize the joint positions as radians.

        Returns:
            The serialized joint positions and the names of the joints.
        """

    @abstractmethod
    def deserialize_joint_positions(
        self,
        names: list[str],
        value: T,
        *,
        radians: bool = False,
    ) -> JointPositionsValue:
        """Deserialize a joint positions value.

        Args:
            names: The names of the joints.
            value: The serialized joint positions.
            radians: Whether the serialized joint positions are radians.

        Returns:
            The deserialized joint positions.
        """


class AudioFrameSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_audio_frame(
        self,
        audio_frame: AudioFrameValue,
        *,
        sample_rate: int | None = None,
    ) -> T:
        """Serialize an audio frame value.

        Args:
            audio_frame: The frame of audio to serialize.
            sample_rate: The target sample rate of the serialized audio frame.

        Returns:
            The serialized audio frame.
        """

    @abstractmethod
    def deserialize_audio_frame(self, value: T, sample_rate: int) -> AudioFrameValue:
        """Deserialize an audio frame value.

        Args:
            value: The serialized audio frame.
            sample_rate: The sample rate of the serialized audio frame.

        Returns:
            The deserialized audio frame.
        """


class CameraFrameSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_camera_frame(self, camera_frame: CameraFrameValue) -> T:
        """Serialize a camera frame value."""

    @abstractmethod
    def deserialize_camera_frame(self, value: T) -> CameraFrameValue:
        """Deserialize a camera frame value."""


class JointTorquesSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_torques(self, joint_torques: JointTorquesValue) -> T:
        """Serialize a joint torques value."""

    @abstractmethod
    def deserialize_joint_torques(self, value: T) -> JointTorquesValue:
        """Deserialize a joint torques value."""


class Serializer(
    TimestampSerializer[T],
    TensorSerializer[T],
    JointPositionsSerializer[T],
    AudioFrameSerializer[T],
    CameraFrameSerializer[T],
    JointTorquesSerializer[T],
    Generic[T],
):
    pass
