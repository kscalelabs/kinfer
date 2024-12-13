"""Defines a serializer for JSON."""

import base64

from kinfer import protos as P
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
)

JsonValue = dict[str, str | float | int | list[str | float | int]]


class JsonJointPositionsSerializer(JointPositionsSerializer[JsonValue]):
    def serialize_joint_positions(
        self,
        schema: P.JointPositionsSchema,
        value: P.JointPositionsValue,
    ) -> dict[str, list[float]]:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        return {
            "positions": [
                convert_angular_position(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ]
        }

    def deserialize_joint_positions(
        self,
        schema: P.JointPositionsSchema,
        value: JsonValue,
    ) -> P.JointPositionsValue:
        if "positions" not in value:
            raise ValueError("Key 'positions' not found in value")
        positions = value["positions"]
        if not isinstance(positions, list) or len(positions) != len(schema.joint_names):
            raise ValueError(
                f"Shape of positions must match number of joint names: {len(positions)} != {len(schema.joint_names)}"
            )
        return P.JointPositionsValue(
            values=[
                P.JointPositionValue(joint_name=name, value=positions[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class JsonJointVelocitiesSerializer(JointVelocitiesSerializer[JsonValue]):
    def serialize_joint_velocities(
        self,
        schema: P.JointVelocitiesSchema,
        value: P.JointVelocitiesValue,
    ) -> dict[str, list[float]]:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        return {
            "velocities": [
                convert_angular_velocity(value_map[name].value, value_map[name].unit, schema.unit)
                for name in schema.joint_names
            ]
        }

    def deserialize_joint_velocities(
        self,
        schema: P.JointVelocitiesSchema,
        value: JsonValue,
    ) -> P.JointVelocitiesValue:
        if "velocities" not in value:
            raise ValueError("Key 'velocities' not found in value")
        velocities = value["velocities"]
        if not isinstance(velocities, list) or len(velocities) != len(schema.joint_names):
            raise ValueError(
                f"Shape of velocities must match number of joint names: {len(velocities)} != {len(schema.joint_names)}"
            )
        return P.JointVelocitiesValue(
            values=[
                P.JointVelocityValue(joint_name=name, value=velocities[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class JsonJointTorquesSerializer(JointTorquesSerializer[JsonValue]):
    def serialize_joint_torques(
        self,
        schema: P.JointTorquesSchema,
        value: P.JointTorquesValue,
    ) -> dict[str, list[float]]:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        return {
            "torques": [
                convert_torque(value_map[name].value, value_map[name].unit, schema.unit) for name in schema.joint_names
            ]
        }

    def deserialize_joint_torques(
        self,
        schema: P.JointTorquesSchema,
        value: JsonValue,
    ) -> P.JointTorquesValue:
        if "torques" not in value:
            raise ValueError("Key 'torques' not found in value")
        torques = value["torques"]
        if not isinstance(torques, list) or len(torques) != len(schema.joint_names):
            raise ValueError(
                f"Shape of torques must match number of joint names: {len(torques)} != {len(schema.joint_names)}"
            )
        return P.JointTorquesValue(
            values=[
                P.JointTorqueValue(joint_name=name, value=torques[i], unit=schema.unit)
                for i, name in enumerate(schema.joint_names)
            ]
        )


class JsonJointCommandsSerializer(JointCommandsSerializer[JsonValue]):
    def _convert_value_to_array(
        self,
        value: P.JointCommandValue,
        schema: P.JointCommandsSchema,
    ) -> dict[str, list[float]]:
        return {
            "command": [
                convert_torque(value.torque, value.torque_unit, schema.torque_unit),
                convert_angular_velocity(value.velocity, value.velocity_unit, schema.velocity_unit),
                convert_angular_position(value.position, value.position_unit, schema.position_unit),
                value.kp,
                value.kd,
            ]
        }

    def _convert_array_to_value(
        self,
        values: JsonValue,
        schema: P.JointCommandsSchema,
        name: str,
    ) -> P.JointCommandValue:
        if "command" not in values:
            raise ValueError("Key 'command' not found in value")
        command = values["command"]
        if not isinstance(command, list) or len(command) != 5:
            raise ValueError(f"Shape of command must match number of joint commands: {len(command)} != 5")
        return P.JointCommandValue(
            joint_name=name,
            torque=command[0],
            velocity=command[1],
            position=command[2],
            kp=command[3],
            kd=command[4],
            torque_unit=schema.torque_unit,
            velocity_unit=schema.velocity_unit,
            position_unit=schema.position_unit,
        )

    def serialize_joint_commands(self, schema: P.JointCommandsSchema, value: P.JointCommandsValue) -> JsonValue:
        value_map = {v.joint_name: v for v in value.values}
        check_names_match("schema", schema.joint_names, "value", value_map.keys())
        return {"commands": [self._convert_value_to_array(value_map[name], schema) for name in schema.joint_names]}

    def deserialize_joint_commands(self, schema: P.JointCommandsSchema, value: JsonValue) -> P.JointCommandsValue:
        if "commands" not in value:
            raise ValueError("Key 'commands' not found in value")
        commands = value["commands"]
        if not isinstance(commands, list) or len(commands) != len(schema.joint_names):
            raise ValueError(
                f"Shape of commands must match number of joint names: {len(commands)} != {len(schema.joint_names)}"
            )
        return P.JointCommandsValue(
            values=[
                self._convert_array_to_value(commands[i], schema, name) for i, name in enumerate(schema.joint_names)
            ]
        )


class JsonCameraFrameSerializer(CameraFrameSerializer[JsonValue]):
    def serialize_camera_frame(self, schema: P.CameraFrameSchema, value: P.CameraFrameValue) -> JsonValue:
        return {"data": base64.b64encode(value.data).decode("utf-8")}

    def deserialize_camera_frame(self, schema: P.CameraFrameSchema, value: JsonValue) -> P.CameraFrameValue:
        if "data" not in value:
            raise ValueError("Key 'data' not found in value")
        data = value["data"]
        if not isinstance(data, str):
            raise ValueError("Key 'data' must be a string")
        return P.CameraFrameValue(data=base64.b64decode(data))


class JsonAudioFrameSerializer(AudioFrameSerializer[JsonValue]):
    def serialize_audio_frame(self, schema: P.AudioFrameSchema, value: P.AudioFrameValue) -> JsonValue:
        return {"data": base64.b64encode(value.data).decode("utf-8")}

    def deserialize_audio_frame(self, schema: P.AudioFrameSchema, value: JsonValue) -> P.AudioFrameValue:
        if "data" not in value:
            raise ValueError("Key 'data' not found in value")
        data = value["data"]
        if not isinstance(data, str):
            raise ValueError("Key 'data' must be a string")
        return P.AudioFrameValue(data=base64.b64decode(data))


class JsonIMUSerializer(IMUSerializer[JsonValue]):
    def serialize_imu(self, schema: P.IMUSchema, value: P.IMUValue) -> JsonValue:
        data: dict[str, list[float]] = {}
        if schema.use_accelerometer:
            data["linear_acceleration"] = [
                value.linear_acceleration.x,
                value.linear_acceleration.y,
                value.linear_acceleration.z,
            ]
        if schema.use_gyroscope:
            data["angular_velocity"] = [value.angular_velocity.x, value.angular_velocity.y, value.angular_velocity.z]
        if schema.use_magnetometer:
            data["magnetic_field"] = [value.magnetic_field.x, value.magnetic_field.y, value.magnetic_field.z]
        return data

    def deserialize_imu(self, schema: P.IMUSchema, value: JsonValue) -> P.IMUValue:
        imu_value = P.IMUValue()
        if schema.use_accelerometer:
            x, y, z = value["linear_acceleration"]
            imu_value.linear_acceleration.x = x
            imu_value.linear_acceleration.y = y
            imu_value.linear_acceleration.z = z
        if schema.use_gyroscope:
            x, y, z = value["angular_velocity"]
            imu_value.angular_velocity.x = x
            imu_value.angular_velocity.y = y
            imu_value.angular_velocity.z = z
        if schema.use_magnetometer:
            x, y, z = value["magnetic_field"]
            imu_value.magnetic_field.x = x
            imu_value.magnetic_field.y = y
            imu_value.magnetic_field.z = z
        return imu_value


class JsonTimestampSerializer(TimestampSerializer[JsonValue]):
    def serialize_timestamp(self, schema: P.TimestampSchema, value: P.TimestampValue) -> JsonValue:
        return {"seconds": value.seconds, "nanos": value.nanos}

    def deserialize_timestamp(self, schema: P.TimestampSchema, value: JsonValue) -> P.TimestampValue:
        return P.TimestampValue(seconds=value["seconds"], nanos=value["nanos"])


class JsonVectorCommandSerializer(VectorCommandSerializer[JsonValue]):
    def serialize_vector_command(self, schema: P.VectorCommandSchema, value: P.VectorCommandValue) -> JsonValue:
        return {"values": list(value.values)}

    def deserialize_vector_command(self, schema: P.VectorCommandSchema, value: JsonValue) -> P.VectorCommandValue:
        if "values" not in value:
            raise ValueError("Key 'values' not found in value")
        values = value["values"]
        if not isinstance(values, list):
            raise ValueError("Key 'values' must be a list")
        return P.VectorCommandValue(values=values)


class JsonStateTensorSerializer(StateTensorSerializer[JsonValue]):
    def serialize_state_tensor(self, schema: P.StateTensorSchema, value: P.StateTensorValue) -> JsonValue:
        return {"data": base64.b64encode(value.data).decode("utf-8")}

    def deserialize_state_tensor(self, schema: P.StateTensorSchema, value: JsonValue) -> P.StateTensorValue:
        if "data" not in value:
            raise ValueError("Key 'data' not found in value")
        data = value["data"]
        if not isinstance(data, str):
            raise ValueError("Key 'data' must be a string")
        return P.StateTensorValue(data=base64.b64decode(data))


class JsonSerializer(
    JsonJointPositionsSerializer,
    JsonJointVelocitiesSerializer,
    JsonJointTorquesSerializer,
    JsonJointCommandsSerializer,
    JsonCameraFrameSerializer,
    JsonAudioFrameSerializer,
    JsonIMUSerializer,
    JsonTimestampSerializer,
    JsonVectorCommandSerializer,
    JsonStateTensorSerializer,
    Serializer[JsonValue],
):
    def __init__(self, schema: P.ValueSchema) -> None:
        Serializer.__init__(self, schema=schema)


class JsonMultiSerializer(MultiSerializer[JsonValue]):
    def __init__(self, schema: P.InputSchema | P.OutputSchema) -> None:
        super().__init__([JsonSerializer(schema=s) for s in schema.inputs])
