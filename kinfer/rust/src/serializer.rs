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
        schema: &ImuSchema,
        value: Value,
    ) -> Result<ImuValue, Box<dyn std::error::Error>>;
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
