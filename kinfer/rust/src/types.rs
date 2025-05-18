use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelMetadata {
    pub joint_names: Vec<String>,
    pub num_commands: Option<usize>,
    pub carry_size: usize,
}

impl ModelMetadata {
    pub fn model_validate_json(json: String) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_str(&json)?)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum InputType {
    JointAngles,
    JointAngularVelocities,
    ProjectedGravity,
    Accelerometer,
    Gyroscope,
    Command,
    Time,
    Carry,
}

impl InputType {
    pub fn get_name(&self) -> &str {
        match self {
            InputType::JointAngles => "joint_angles",
            InputType::JointAngularVelocities => "joint_angular_velocities",
            InputType::ProjectedGravity => "projected_gravity",
            InputType::Accelerometer => "accelerometer",
            InputType::Gyroscope => "gyroscope",
            InputType::Command => "command",
            InputType::Time => "time",
            InputType::Carry => "carry",
        }
    }

    pub fn get_shape(&self, metadata: &ModelMetadata) -> Vec<usize> {
        match self {
            InputType::JointAngles => vec![metadata.joint_names.len()],
            InputType::JointAngularVelocities => vec![metadata.joint_names.len()],
            InputType::ProjectedGravity => vec![3],
            InputType::Accelerometer => vec![3],
            InputType::Gyroscope => vec![3],
            InputType::Command => vec![metadata.num_commands.unwrap()],
            InputType::Time => vec![1],
            InputType::Carry => vec![metadata.carry_size],
        }
    }

    pub fn from_name(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match name {
            "joint_angles" => Ok(InputType::JointAngles),
            "joint_angular_velocities" => Ok(InputType::JointAngularVelocities),
            "projected_gravity" => Ok(InputType::ProjectedGravity),
            "accelerometer" => Ok(InputType::Accelerometer),
            "gyroscope" => Ok(InputType::Gyroscope),
            "command" => Ok(InputType::Command),
            "time" => Ok(InputType::Time),
            "carry" => Ok(InputType::Carry),
            _ => Err(format!("Unknown input type: {}", name).into()),
        }
    }
}
