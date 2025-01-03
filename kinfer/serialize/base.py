"""Defines functions for serializing and deserializing signatures."""

from abc import ABC, abstractmethod
from typing import Generic, Literal, Sequence, TypeVar, overload

from kinfer import proto as K

T = TypeVar("T")


class JointPositionsSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_positions(
        self: "JointPositionsSerializer[T]",
        schema: K.JointPositionsSchema,
        value: K.JointPositionsValue,
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
        self: "JointPositionsSerializer[T]",
        schema: K.JointPositionsSchema,
        value: T,
    ) -> K.JointPositionsValue:
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
        self: "JointVelocitiesSerializer[T]",
        schema: K.JointVelocitiesSchema,
        value: K.JointVelocitiesValue,
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
        self: "JointVelocitiesSerializer[T]",
        schema: K.JointVelocitiesSchema,
        value: T,
    ) -> K.JointVelocitiesValue:
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
        self: "JointTorquesSerializer[T]",
        schema: K.JointTorquesSchema,
        value: K.JointTorquesValue,
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
        self: "JointTorquesSerializer[T]",
        schema: K.JointTorquesSchema,
        value: T,
    ) -> K.JointTorquesValue:
        """Deserialize a joint torques value.

        Args:
            schema: The schema of the joint torques.
            value: The serialized joint torques.

        Returns:
            The deserialized joint torques.
        """


class JointCommandsSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_commands(
        self: "JointCommandsSerializer[T]",
        schema: K.JointCommandsSchema,
        value: K.JointCommandsValue,
    ) -> T:
        """Serialize a joint commands value.

        Args:
            schema: The schema of the joint commands.
            value: The joint commands to serialize.

        Returns:
            The serialized joint commands.
        """

    @abstractmethod
    def deserialize_joint_commands(
        self: "JointCommandsSerializer[T]",
        schema: K.JointCommandsSchema,
        value: T,
    ) -> K.JointCommandsValue:
        """Deserialize a joint commands value.

        Args:
            schema: The schema of the joint commands.
            value: The serialized joint commands.

        Returns:
            The deserialized joint commands.
        """


class CameraFrameSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_camera_frame(
        self: "CameraFrameSerializer[T]",
        schema: K.CameraFrameSchema,
        value: K.CameraFrameValue,
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
        self: "CameraFrameSerializer[T]",
        schema: K.CameraFrameSchema,
        value: T,
    ) -> K.CameraFrameValue:
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
        self: "AudioFrameSerializer[T]",
        schema: K.AudioFrameSchema,
        value: K.AudioFrameValue,
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
        self: "AudioFrameSerializer[T]",
        schema: K.AudioFrameSchema,
        value: T,
    ) -> K.AudioFrameValue:
        """Deserialize an audio frame value.

        Args:
            schema: The schema of the audio frame.
            value: The serialized audio frame.

        Returns:
            The deserialized audio frame.
        """


class ImuSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_imu(
        self: "ImuSerializer[T]",
        schema: K.ImuSchema,
        value: K.ImuValue,
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
        self: "ImuSerializer[T]",
        schema: K.ImuSchema,
        value: T,
    ) -> K.ImuValue:
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
        self: "TimestampSerializer[T]",
        schema: K.TimestampSchema,
        value: K.TimestampValue,
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
        self: "TimestampSerializer[T]",
        schema: K.TimestampSchema,
        value: T,
    ) -> K.TimestampValue:
        """Deserialize a timestamp value.

        Args:
            schema: The schema of the timestamp.
            value: The serialized timestamp.

        Returns:
            The deserialized timestamp.
        """


class VectorCommandSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_vector_command(
        self: "VectorCommandSerializer[T]",
        schema: K.VectorCommandSchema,
        value: K.VectorCommandValue,
    ) -> T:
        """Serialize an XY command value.

        Args:
            schema: The schema of the vector command.
            value: The vector command to serialize.

        Returns:
            The serialized vector command.
        """

    @abstractmethod
    def deserialize_vector_command(
        self: "VectorCommandSerializer[T]",
        schema: K.VectorCommandSchema,
        value: T,
    ) -> K.VectorCommandValue:
        """Deserialize a vector command value.

        Args:
            schema: The schema of the vector command.
            value: The serialized vector command.

        Returns:
            The deserialized vector command.
        """


class StateTensorSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_state_tensor(
        self: "StateTensorSerializer[T]",
        schema: K.StateTensorSchema,
        value: K.StateTensorValue,
    ) -> T:
        """Serialize a state tensor value.

        Args:
            schema: The schema of the state.
            value: The state to serialize.

        Returns:
            The serialized state.
        """

    @abstractmethod
    def deserialize_state_tensor(
        self: "StateTensorSerializer[T]",
        schema: K.StateTensorSchema,
        value: T,
    ) -> K.StateTensorValue:
        """Deserialize a state tensor value.

        Args:
            schema: The schema of the state.
            value: The serialized state.

        Returns:
            The deserialized state.
        """


class Serializer(
    JointPositionsSerializer[T],
    JointVelocitiesSerializer[T],
    JointTorquesSerializer[T],
    JointCommandsSerializer[T],
    CameraFrameSerializer[T],
    AudioFrameSerializer[T],
    ImuSerializer[T],
    TimestampSerializer[T],
    VectorCommandSerializer[T],
    StateTensorSerializer[T],
    Generic[T],
):
    def __init__(self: "Serializer[T]", schema: K.ValueSchema) -> None:
        self.schema = schema

    def serialize(self: "Serializer[T]", value: K.Value) -> T:
        value_type = value.WhichOneof("value")

        match value_type:
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
            case "joint_commands":
                return self.serialize_joint_commands(
                    schema=self.schema.joint_commands,
                    value=value.joint_commands,
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
            case "vector_command":
                return self.serialize_vector_command(
                    schema=self.schema.vector_command,
                    value=value.vector_command,
                )
            case "state_tensor":
                return self.serialize_state_tensor(
                    schema=self.schema.state_tensor,
                    value=value.state_tensor,
                )
            case _:
                raise ValueError(f"Unsupported value type: {value_type}")

    def deserialize(self: "Serializer[T]", value: T) -> K.Value:
        value_type = self.schema.WhichOneof("value_type")

        match value_type:
            case "joint_positions":
                return K.Value(
                    joint_positions=self.deserialize_joint_positions(
                        schema=self.schema.joint_positions,
                        value=value,
                    ),
                )
            case "joint_velocities":
                return K.Value(
                    joint_velocities=self.deserialize_joint_velocities(
                        schema=self.schema.joint_velocities,
                        value=value,
                    ),
                )
            case "joint_torques":
                return K.Value(
                    joint_torques=self.deserialize_joint_torques(
                        schema=self.schema.joint_torques,
                        value=value,
                    ),
                )
            case "joint_commands":
                return K.Value(
                    joint_commands=self.deserialize_joint_commands(
                        schema=self.schema.joint_commands,
                        value=value,
                    ),
                )
            case "camera_frame":
                return K.Value(
                    camera_frame=self.deserialize_camera_frame(
                        schema=self.schema.camera_frame,
                        value=value,
                    ),
                )
            case "audio_frame":
                return K.Value(
                    audio_frame=self.deserialize_audio_frame(
                        schema=self.schema.audio_frame,
                        value=value,
                    ),
                )
            case "imu":
                return K.Value(
                    imu=self.deserialize_imu(
                        schema=self.schema.imu,
                        value=value,
                    ),
                )
            case "timestamp":
                return K.Value(
                    timestamp=self.deserialize_timestamp(
                        schema=self.schema.timestamp,
                        value=value,
                    ),
                )
            case "vector_command":
                return K.Value(
                    vector_command=self.deserialize_vector_command(
                        schema=self.schema.vector_command,
                        value=value,
                    ),
                )
            case "state_tensor":
                return K.Value(
                    state_tensor=self.deserialize_state_tensor(
                        schema=self.schema.state_tensor,
                        value=value,
                    ),
                )
            case _:
                raise ValueError(f"Unsupported value type: {value_type}")


class MultiSerializer(Generic[T]):
    def __init__(self: "MultiSerializer[T]", serializers: Sequence[Serializer[T]]) -> None:
        self.serializers = list(serializers)

    @overload
    def serialize_io(self: "MultiSerializer[T]", io: K.IO, *, as_dict: Literal[True]) -> dict[str, T]: ...

    @overload
    def serialize_io(self: "MultiSerializer[T]", io: K.IO, *, as_dict: Literal[False] = False) -> list[T]: ...

    def serialize_io(self: "MultiSerializer[T]", io: K.IO, *, as_dict: bool = False) -> dict[str, T] | list[T]:
        if not isinstance(io, K.IO):
            raise ValueError(f"Inputs must be an IO protobuf, not {type(io)}")
        if as_dict:
            return {s.schema.value_name: s.serialize(i) for s, i in zip(self.serializers, io.values)}
        return [s.serialize(i) for s, i in zip(self.serializers, io.values)]

    def deserialize_io(self: "MultiSerializer[T]", io: dict[str, T] | list[T]) -> K.IO:
        if not isinstance(io, (dict, list)):
            raise ValueError(f"Inputs must be a dictionary or list, not {type(io)}")
        if isinstance(io, dict):
            return K.IO(values=[s.deserialize(i) for s, i in zip(self.serializers, io.values())])
        return K.IO(values=[s.deserialize(i) for s, i in zip(self.serializers, io)])

    def assign_names(self: "MultiSerializer[T]", values: Sequence[T]) -> dict[str, T]:
        if not isinstance(values, Sequence):
            raise ValueError(f"Values must be a sequence, not {type(values)}")
        if len(values) != len(self.serializers):
            raise ValueError(f"Expected {len(self.serializers)} values, got {len(values)}")
        return {s.schema.value_name: v for s, v in zip(self.serializers, values)}

    @property
    def names(self: "MultiSerializer[T]") -> list[str]:
        return [s.schema.value_name for s in self.serializers]
