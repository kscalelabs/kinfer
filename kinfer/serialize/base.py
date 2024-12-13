"""Defines functions for serializing and deserializing signatures."""

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from kinfer import protos as P

T = TypeVar("T")


class JointPositionsSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_joint_positions(
        self,
        schema: P.JointPositionsSchema,
        value: P.JointPositionsValue,
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
        schema: P.JointPositionsSchema,
        value: T,
    ) -> P.JointPositionsValue:
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
        schema: P.JointVelocitiesSchema,
        value: P.JointVelocitiesValue,
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
        schema: P.JointVelocitiesSchema,
        value: T,
    ) -> P.JointVelocitiesValue:
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
        schema: P.JointTorquesSchema,
        value: P.JointTorquesValue,
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
        schema: P.JointTorquesSchema,
        value: T,
    ) -> P.JointTorquesValue:
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
        self,
        schema: P.JointCommandsSchema,
        value: P.JointCommandsValue,
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
        self,
        schema: P.JointCommandsSchema,
        value: T,
    ) -> P.JointCommandsValue:
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
        self,
        schema: P.CameraFrameSchema,
        value: P.CameraFrameValue,
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
        schema: P.CameraFrameSchema,
        value: T,
    ) -> P.CameraFrameValue:
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
        schema: P.AudioFrameSchema,
        value: P.AudioFrameValue,
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
        schema: P.AudioFrameSchema,
        value: T,
    ) -> P.AudioFrameValue:
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
        schema: P.IMUSchema,
        value: P.IMUValue,
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
        schema: P.IMUSchema,
        value: T,
    ) -> P.IMUValue:
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
        schema: P.TimestampSchema,
        value: P.TimestampValue,
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
        schema: P.TimestampSchema,
        value: T,
    ) -> P.TimestampValue:
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
        self,
        schema: P.VectorCommandSchema,
        value: P.VectorCommandValue,
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
        self,
        schema: P.VectorCommandSchema,
        value: T,
    ) -> P.VectorCommandValue:
        """Deserialize a vector command value.

        Args:
            schema: The schema of the vector command.
            value: The serialized vector command.

        Returns:
            The deserialized vector command.
        """


class StateTensorSerializer(ABC, Generic[T]):
    @abstractmethod
    def serialize_state_tensor(self, schema: P.StateTensorSchema, value: P.StateTensorValue) -> T:
        """Serialize a state tensor value.

        Args:
            schema: The schema of the state.
            value: The state to serialize.

        Returns:
            The serialized state.
        """

    @abstractmethod
    def deserialize_state_tensor(self, schema: P.StateTensorSchema, value: T) -> P.StateTensorValue:
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
    IMUSerializer[T],
    TimestampSerializer[T],
    VectorCommandSerializer[T],
    StateTensorSerializer[T],
    Generic[T],
):
    def __init__(self, schema: P.ValueSchema) -> None:
        self.schema = schema

    def serialize(self, value: P.Value) -> T:
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

    def deserialize(self, value: T) -> P.Value:
        value_type = self.schema.WhichOneof("value_type")

        match value_type:
            case "joint_positions":
                return P.Value(
                    joint_positions=self.deserialize_joint_positions(
                        schema=self.schema.joint_positions,
                        value=value,
                    ),
                )
            case "joint_velocities":
                return P.Value(
                    joint_velocities=self.deserialize_joint_velocities(
                        schema=self.schema.joint_velocities,
                        value=value,
                    ),
                )
            case "joint_torques":
                return P.Value(
                    joint_torques=self.deserialize_joint_torques(
                        schema=self.schema.joint_torques,
                        value=value,
                    ),
                )
            case "joint_commands":
                return P.Value(
                    joint_commands=self.deserialize_joint_commands(
                        schema=self.schema.joint_commands,
                        value=value,
                    ),
                )
            case "camera_frame":
                return P.Value(
                    camera_frame=self.deserialize_camera_frame(
                        schema=self.schema.camera_frame,
                        value=value,
                    ),
                )
            case "audio_frame":
                return P.Value(
                    audio_frame=self.deserialize_audio_frame(
                        schema=self.schema.audio_frame,
                        value=value,
                    ),
                )
            case "imu":
                return P.Value(
                    imu=self.deserialize_imu(
                        schema=self.schema.imu,
                        value=value,
                    ),
                )
            case "timestamp":
                return P.Value(
                    timestamp=self.deserialize_timestamp(
                        schema=self.schema.timestamp,
                        value=value,
                    ),
                )
            case "vector_command":
                return P.Value(
                    vector_command=self.deserialize_vector_command(
                        schema=self.schema.vector_command,
                        value=value,
                    ),
                )
            case "state_tensor":
                return P.Value(
                    state_tensor=self.deserialize_state_tensor(
                        schema=self.schema.state_tensor,
                        value=value,
                    ),
                )
            case _:
                raise ValueError(f"Unsupported value type: {value_type}")


class MultiSerializer(Generic[T]):
    def __init__(self, serializers: Sequence[Serializer[T]]) -> None:
        self.serializers = list(serializers)

    def serialize_input(self, input: P.Input) -> dict[str, T]:
        return {s.schema.value_name: s.serialize(i) for s, i in zip(self.serializers, input.inputs)}

    def serialize_output(self, output: P.Output) -> dict[str, T]:
        return {s.schema.value_name: s.serialize(o) for s, o in zip(self.serializers, output.outputs)}

    def deserialize_input(self, input: dict[str, T]) -> P.Input:
        return P.Input(inputs=[s.deserialize(i) for s, i in zip(self.serializers, input.values())])

    def deserialize_output(self, output: dict[str, T]) -> P.Output:
        return P.Output(outputs=[s.deserialize(o) for s, o in zip(self.serializers, output.values())])
