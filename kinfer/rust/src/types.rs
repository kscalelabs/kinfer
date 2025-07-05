use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelMetadata {
    pub joint_names: Vec<String>,
    pub num_commands: Option<usize>,
    pub carry_size: Vec<usize>,
    pub joint_biases: Option<Vec<JointBias>>,
    pub command_type_info: Option<CommandTypeInfo>,
    pub kinfer_version: Option<String>,
    pub training_metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct JointBias {
    pub joint_name: String,
    pub reference_angle: f64,
    pub weight: f64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CommandTypeInfo {
    pub command_type: String,
    pub description: String,
    pub fields: Vec<CommandField>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CommandField {
    pub name: String,
    pub description: String,
    pub units: Option<String>,
    pub range: Option<(f64, f64)>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TransportEnvelope {
    pub command_type: String,
    pub payload_type: PayloadType,
    pub payload_length: Option<usize>,
    pub codec_info: Option<CodecInfo>,
    pub kinfer_version: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum PayloadType {
    FloatVector,
    Text,
    Audio,
    Image,
    Proto,
    Binary,
    Json,
    Custom(String),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CodecInfo {
    pub format: String,
    pub sample_rate: Option<u32>,     // For audio
    pub channels: Option<u8>,         // For audio
    pub width: Option<u32>,           // For image
    pub height: Option<u32>,          // For image
    pub encoding: Option<String>,     // For text/proto
    pub compression: Option<String>,  // For any type
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PayloadSchema {
    pub schema_type: PayloadType,
    pub schema_spec: SchemaSpec,
    pub description: String,
    pub version: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum SchemaSpec {
    FloatVector(FloatVectorSchema),
    Text(TextSchema),
    Audio(AudioSchema),
    Image(ImageSchema),
    Proto(ProtoSchema),
    Binary(BinarySchema),
    Json(JsonSchema),
    Custom(CustomSchema),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FloatVectorSchema {
    pub fields: Vec<CommandField>,
    pub total_length: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TextSchema {
    pub grammar: Option<String>,
    pub json_schema: Option<String>,
    pub max_length: Option<usize>,
    pub encoding: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AudioSchema {
    pub sample_rate: u32,
    pub channels: u8,
    pub bit_depth: u8,
    pub format: String,
    pub max_duration_ms: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageSchema {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub pixel_format: String,
    pub compression: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProtoSchema {
    pub proto_definition: String,
    pub message_type: String,
    pub version: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BinarySchema {
    pub max_length: Option<usize>,
    pub structure_definition: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct JsonSchema {
    pub schema: String,
    pub version: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CustomSchema {
    pub schema_definition: String,
    pub decoder_info: HashMap<String, serde_json::Value>,
}


impl ModelMetadata {
    pub fn model_validate_json(json: String) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_str(&json)?)
    }

    pub fn to_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn get_joint_bias_by_name(&self, joint_name: &str) -> Option<&JointBias> {
        self.joint_biases.as_ref()?.iter().find(|bias| bias.joint_name == joint_name)
    }

    pub fn get_command_type(&self) -> Option<&str> {
        self.command_type_info.as_ref().map(|info| info.command_type.as_str())
    }

    pub fn get_command_description(&self) -> Option<&str> {
        self.command_type_info.as_ref().map(|info| info.description.as_str())
    }

    pub fn get_transport_envelope(&self) -> Option<TransportEnvelope> {
        self.command_type_info.as_ref().map(|info| {
            let payload_type = if info.fields.is_empty() {
                PayloadType::Custom("pose".to_string())
            } else {
                PayloadType::FloatVector
            };
            
            TransportEnvelope {
                command_type: info.command_type.clone(),
                payload_type,
                payload_length: self.num_commands,
                codec_info: None,
                kinfer_version: self.kinfer_version.clone(),
            }
        })
    }

    pub fn validate_transport_compatibility(&self, expected_envelope: &TransportEnvelope) -> Result<(), String> {
        let actual_envelope = self.get_transport_envelope()
            .ok_or("No transport envelope available in model metadata")?;
        
        if actual_envelope.command_type != expected_envelope.command_type {
            return Err(format!(
                "Command type mismatch: expected '{}', found '{}'",
                expected_envelope.command_type, actual_envelope.command_type
            ));
        }
        
        if !payload_types_compatible(&actual_envelope.payload_type, &expected_envelope.payload_type) {
            return Err(format!(
                "Payload type mismatch: expected '{:?}', found '{:?}'",
                expected_envelope.payload_type, actual_envelope.payload_type
            ));
        }
        
        Ok(())
    }
    
    pub fn validate_command_compatibility(&self, expected_command_type: &str) -> Result<(), String> {
        match self.get_command_type() {
            Some(actual_type) if actual_type == expected_command_type => Ok(()),
            Some(actual_type) => {
                let description = self.get_command_description().unwrap_or("No description available");
                Err(format!(
                    "Command type mismatch: expected '{}', found '{}'\n\
                     Command structure for '{}':\n{}\n\
                     Fields: {:?}",
                    expected_command_type, actual_type, actual_type, description,
                    self.command_type_info.as_ref().map(|info| &info.fields).unwrap_or(&vec![])
                ))
            }
            None => Err(format!(
                "No command type information available in model metadata. \
                 Expected command type: '{}'",
                expected_command_type
            ))
        }
    }
}

fn payload_types_compatible(actual: &PayloadType, expected: &PayloadType) -> bool {
    match (actual, expected) {
        (PayloadType::FloatVector, PayloadType::FloatVector) => true,
        (PayloadType::Text, PayloadType::Text) => true,
        (PayloadType::Audio, PayloadType::Audio) => true,
        (PayloadType::Image, PayloadType::Image) => true,
        (PayloadType::Proto, PayloadType::Proto) => true,
        (PayloadType::Binary, PayloadType::Binary) => true,
        (PayloadType::Json, PayloadType::Json) => true,
        (PayloadType::Custom(a), PayloadType::Custom(b)) => a == b,
        _ => false,
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum InputType {
    JointAngles,
    JointAngularVelocities,
    InitialHeading,
    Quaternion,
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
            InputType::InitialHeading => "initial_heading",
            InputType::Quaternion => "quaternion",
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
            InputType::InitialHeading => vec![1],
            InputType::Quaternion => vec![4],
            InputType::ProjectedGravity => vec![3],
            InputType::Accelerometer => vec![3],
            InputType::Gyroscope => vec![3],
            InputType::Command => vec![metadata.num_commands.unwrap_or(0)],
            InputType::Time => vec![1],
            InputType::Carry => metadata.carry_size.clone(),
        }
    }

    pub fn from_name(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match name {
            "joint_angles" => Ok(InputType::JointAngles),
            "joint_angular_velocities" => Ok(InputType::JointAngularVelocities),
            "initial_heading" => Ok(InputType::InitialHeading),
            "quaternion" => Ok(InputType::Quaternion),
            "projected_gravity" => Ok(InputType::ProjectedGravity),
            "accelerometer" => Ok(InputType::Accelerometer),
            "gyroscope" => Ok(InputType::Gyroscope),
            "command" => Ok(InputType::Command),
            "time" => Ok(InputType::Time),
            "carry" => Ok(InputType::Carry),
            _ => Err(format!("Unknown input type: {}", name).into()),
        }
    }

    pub fn get_names() -> Vec<&'static str> {
        vec![
            "joint_angles",
            "joint_angular_velocities",
            "initial_heading",
            "quaternion",
            "projected_gravity",
            "accelerometer",
            "gyroscope",
            "command",
            "time",
            "carry",
        ]
    }
}
