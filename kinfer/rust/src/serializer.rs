use crate::kinfer_proto::{
    AudioFrameSchema, AudioFrameValue, CameraFrameSchema, CameraFrameValue, ImuSchema, ImuValue,
    JointCommandsSchema, JointCommandsValue, JointPositionUnit, JointPositionsSchema,
    JointPositionsValue, JointTorqueUnit, JointTorquesSchema, JointTorquesValue,
    JointVelocitiesSchema, JointVelocitiesValue, JointVelocityUnit, ProtoIO, ProtoIOSchema,
    ProtoValue, StateTensorSchema, StateTensorValue, TimestampSchema, TimestampValue, ValueSchema,
    VectorCommandSchema, VectorCommandValue,
};

use ort::value::Value as OrtValue;
use std::error::Error;

pub trait JointPositionsSerializer {
    fn serialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: JointPositionsValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: OrtValue,
    ) -> Result<JointPositionsValue, Box<dyn Error>>;
}

pub trait JointVelocitiesSerializer {
    fn serialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: JointVelocitiesValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: OrtValue,
    ) -> Result<JointVelocitiesValue, Box<dyn Error>>;
}

pub trait JointTorquesSerializer {
    fn serialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: JointTorquesValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: OrtValue,
    ) -> Result<JointTorquesValue, Box<dyn Error>>;
}

pub trait JointCommandsSerializer {
    fn serialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: JointCommandsValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: OrtValue,
    ) -> Result<JointCommandsValue, Box<dyn Error>>;
}

pub trait CameraFrameSerializer {
    fn serialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: CameraFrameValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: OrtValue,
    ) -> Result<CameraFrameValue, Box<dyn Error>>;
}

pub trait AudioFrameSerializer {
    fn serialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: AudioFrameValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: OrtValue,
    ) -> Result<AudioFrameValue, Box<dyn Error>>;
}

pub trait ImuSerializer {
    fn serialize_imu(
        &self,
        schema: &ImuSchema,
        value: ImuValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_imu(
        &self,
        schema: &ImuSchema,
        value: OrtValue,
    ) -> Result<ImuValue, Box<dyn Error>>;
}

pub trait TimestampSerializer {
    fn serialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: TimestampValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: OrtValue,
    ) -> Result<TimestampValue, Box<dyn Error>>;
}

pub trait VectorCommandSerializer {
    fn serialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: VectorCommandValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: OrtValue,
    ) -> Result<VectorCommandValue, Box<dyn Error>>;
}

pub trait StateTensorSerializer {
    fn serialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: StateTensorValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: OrtValue,
    ) -> Result<StateTensorValue, Box<dyn Error>>;
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
        value: ProtoValue,
    ) -> Result<OrtValue, Box<dyn Error>>;

    fn deserialize(
        &self,
        schema: &ValueSchema,
        value: OrtValue,
    ) -> Result<ProtoValue, Box<dyn Error>>;
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
