use crate::proto::*;
use ndarray::Array1;
use ort::Value;
use std::collections::HashMap;

pub trait JointPositionsSerializer {
    fn serialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: JointPositionsValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: Value,
    ) -> Result<JointPositionsValue, Box<dyn std::error::Error>>;
}

pub trait JointVelocitiesSerializer {
    fn serialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: JointVelocitiesValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: Value,
    ) -> Result<JointVelocitiesValue, Box<dyn std::error::Error>>;
}

pub trait JointTorquesSerializer {
    fn serialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: JointTorquesValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: Value,
    ) -> Result<JointTorquesValue, Box<dyn std::error::Error>>;
}

pub trait JointCommandsSerializer {
    fn serialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: JointCommandsValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: Value,
    ) -> Result<JointCommandsValue, Box<dyn std::error::Error>>;
}

pub trait CameraFrameSerializer {
    fn serialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: CameraFrameValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: Value,
    ) -> Result<CameraFrameValue, Box<dyn std::error::Error>>;
}

pub trait AudioFrameSerializer {
    fn serialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: AudioFrameValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: Value,
    ) -> Result<AudioFrameValue, Box<dyn std::error::Error>>;
}

pub trait ImuSerializer {
    fn serialize_imu(
        &self,
        schema: &ImuSchema,
        value: ImuValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_imu(
        &self,
        schema: &IMUSchema,
        value: Value,
    ) -> Result<IMUValue, Box<dyn std::error::Error>>;
}

pub trait TimestampSerializer {
    fn serialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: TimestampValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: Value,
    ) -> Result<TimestampValue, Box<dyn std::error::Error>>;
}

pub trait VectorCommandSerializer {
    fn serialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: VectorCommandValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: Value,
    ) -> Result<VectorCommandValue, Box<dyn std::error::Error>>;
}

pub trait StateTensorSerializer {
    fn serialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: StateTensorValue,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: Value,
    ) -> Result<StateTensorValue, Box<dyn std::error::Error>>;
}

pub trait Serializer:
    JointPositionsSerializer
    + JointVelocitiesSerializer
    + JointTorquesSerializer
    + JointCommandsSerializer
    + CameraFrameSerializer
    + AudioFrameSerializer
    + ImuSerializer
    + TimestampSerializer
    + VectorCommandSerializer
    + StateTensorSerializer
{
    fn serialize(
        &self,
        schema: &ValueSchema,
        value: Value,
    ) -> Result<Value, Box<dyn std::error::Error>>;

    fn deserialize(
        &self,
        schema: &ValueSchema,
        value: Value,
    ) -> Result<Value, Box<dyn std::error::Error>>;
}

pub fn convert_position(
    value: f32,
    from_unit: JointPositionUnit,
    to_unit: JointPositionUnit,
) -> Result<f32, Box<dyn Error>> {
    match (from_unit, to_unit) {
        (JointPositionUnit::Radians, JointPositionUnit::Degrees) => {
            Ok(value * 180.0 / std::f32::consts::PI)
        }
        (JointPositionUnit::Degrees, JointPositionUnit::Radians) => {
            Ok(value * std::f32::consts::PI / 180.0)
        }
        (a, b) if a == b => Ok(value),
        _ => Err("Unsupported position unit conversion".into()),
    }
}

pub fn convert_velocity(
    value: f32,
    from_unit: JointVelocityUnit,
    to_unit: JointVelocityUnit,
) -> Result<f32, Box<dyn Error>> {
    match (from_unit, to_unit) {
        (JointVelocityUnit::RadiansPerSecond, JointVelocityUnit::DegreesPerSecond) => {
            Ok(value * 180.0 / std::f32::consts::PI)
        }
        (JointVelocityUnit::DegreesPerSecond, JointVelocityUnit::RadiansPerSecond) => {
            Ok(value * std::f32::consts::PI / 180.0)
        }
        (a, b) if a == b => Ok(value),
        _ => Err("Unsupported velocity unit conversion".into()),
    }
}

pub fn convert_torque(
    value: f32,
    from_unit: JointTorqueUnit,
    to_unit: JointTorqueUnit,
) -> Result<f32, Box<dyn Error>> {
    match (from_unit, to_unit) {
        (a, b) if a == b => Ok(value),
        _ => Err("Unsupported torque unit conversion".into()),
    }
}
