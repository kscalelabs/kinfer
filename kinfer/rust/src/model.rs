use async_trait::async_trait;
use flate2::read::GzDecoder;
use futures_util::future;
use ndarray::{Array, IxDyn};
use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use tar::Archive;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    joint_names: Vec<String>,
    num_commands: Option<usize>,
}

impl ModelMetadata {
    fn model_validate_json(json: String) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_str(&json)?)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Provider error: {0}")]
    Provider(String),
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn get_joint_angles(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn get_joint_angular_velocities(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn get_projected_gravity(&self) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn get_accelerometer(&self) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn get_gyroscope(&self) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn get_command(&self) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn get_carry(&self, carry: Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>, ModelError>;
    async fn take_action(
        &self,
        joint_names: Vec<String>,
        action: Array<f32, IxDyn>,
    ) -> Result<(), ModelError>;
}

pub struct ModelRunner {
    init_session: Session,
    step_session: Session,
    metadata: ModelMetadata,
    provider: Arc<dyn ModelProvider>,
}

impl ModelRunner {
    pub async fn new<P: AsRef<Path>>(
        model_path: P,
        input_provider: Arc<dyn ModelProvider>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(model_path).await?;

        // Read entire file into memory
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;

        // Decompress and read the tar archive from memory
        let gz = GzDecoder::new(&buffer[..]);
        let mut archive = Archive::new(gz);

        // Extract and validate joint names
        let mut metadata: Option<String> = None;
        let mut init_fn: Option<Vec<u8>> = None;
        let mut step_fn: Option<Vec<u8>> = None;

        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            let path_str = path.to_string_lossy();

            match path_str.as_ref() {
                "metadata.json" => {
                    let mut contents = String::new();
                    entry.read_to_string(&mut contents)?;
                    metadata = Some(contents);
                }
                "init_fn.onnx" => {
                    let size = entry.size() as usize;
                    let mut contents = vec![0u8; size];
                    entry.read_exact(&mut contents)?;
                    assert_eq!(contents.len(), entry.size() as usize);
                    init_fn = Some(contents);
                }
                "step_fn.onnx" => {
                    let size = entry.size() as usize;
                    let mut contents = vec![0u8; size];
                    entry.read_exact(&mut contents)?;
                    assert_eq!(contents.len(), entry.size() as usize);
                    step_fn = Some(contents);
                }
                _ => return Err("Unknown entry".into()),
            }
        }

        // Reads the files.
        let metadata = ModelMetadata::model_validate_json(
            metadata.ok_or("metadata.json not found in archive")?,
        )?;
        let init_session = Session::builder()?
            .commit_from_memory(&init_fn.ok_or("init_fn.onnx not found in archive")?)?;
        let step_session = Session::builder()?
            .commit_from_memory(&step_fn.ok_or("step_fn.onnx not found in archive")?)?;

        // Validate init_fn has no inputs and one output
        if !init_session.inputs.is_empty() {
            return Err("init_fn should not have any inputs".into());
        }
        if init_session.outputs.len() != 1 {
            return Err("init_fn should have exactly one output".into());
        }

        // Get carry shape from init_fn output
        let carry_shape = init_session.outputs[0]
            .output_type
            .tensor_dimensions()
            .ok_or("Missing tensor type")?
            .to_vec();

        // Validate step_fn inputs and outputs
        Self::validate_step_fn(&step_session, &metadata, &carry_shape)?;

        Ok(Self {
            init_session,
            step_session,
            metadata,
            provider: input_provider,
        })
    }

    fn validate_step_fn(
        session: &Session,
        metadata: &ModelMetadata,
        carry_shape: &[i64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Validate inputs
        for input in &session.inputs {
            let dims = input.input_type.tensor_dimensions().ok_or(format!(
                "Input {} is not a tensor with known dimensions",
                input.name
            ))?;

            match input.name.as_str() {
                "joint_angles" | "joint_angular_velocities" => {
                    let num_joints = metadata.joint_names.len();
                    if *dims != vec![num_joints as i64] {
                        return Err(format!(
                            "Expected shape [{num_joints}] for input `{}`, got {:?}",
                            input.name, dims
                        )
                        .into());
                    }
                }
                "projected_gravity" | "accelerometer" | "gyroscope" => {
                    if *dims != vec![3] {
                        return Err(format!(
                            "Expected shape [3] for input `{}`, got {:?}",
                            input.name, dims
                        )
                        .into());
                    }
                }
                "command" => {
                    let num_commands = metadata.num_commands.ok_or("num_commands is not set")?;
                    if *dims != vec![num_commands as i64] {
                        return Err(format!(
                            "Expected shape [{num_commands}] for input `{}`, got {:?}",
                            input.name, dims
                        )
                        .into());
                    }
                }
                "carry" => {
                    if dims != carry_shape {
                        return Err(format!(
                            "Expected shape {:?} for input `carry`, got {:?}",
                            carry_shape, dims
                        )
                        .into());
                    }
                }
                _ => return Err(format!("Unknown input name: {}", input.name).into()),
            }
        }

        // Validate outputs
        if session.outputs.len() != 2 {
            return Err("Step function must have exactly 2 outputs".into());
        }

        let output_shape = session.outputs[0]
            .output_type
            .tensor_dimensions()
            .ok_or("Missing tensor type")?;
        let num_joints = metadata.joint_names.len();
        if *output_shape != vec![num_joints as i64] {
            return Err(format!(
                "Expected output shape [{num_joints}], got {:?}",
                output_shape
            )
            .into());
        }

        let infered_carry_shape = session.outputs[1]
            .output_type
            .tensor_dimensions()
            .ok_or("Missing tensor type")?;
        if *infered_carry_shape != *carry_shape {
            return Err(format!(
                "Expected carry shape {:?}, got {:?}",
                carry_shape, infered_carry_shape
            )
            .into());
        }

        Ok(())
    }

    pub async fn init(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let input_values: Vec<(&str, Value)> = Vec::new();
        let outputs = self.init_session.run(input_values)?;
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        Ok(output_tensor.view().to_owned())
    }

    pub async fn step(
        &self,
        carry: Array<f32, IxDyn>,
    ) -> Result<(Array<f32, IxDyn>, Array<f32, IxDyn>), Box<dyn std::error::Error>> {
        // Gets the model input names.
        let input_names: Vec<String> = self
            .step_session
            .inputs
            .iter()
            .map(|i| i.name.clone())
            .collect();

        // Calls the relevant getter methods in parallel.
        let mut futures = Vec::new();
        for name in &input_names {
            match name.as_str() {
                "joint_angles" => {
                    futures.push(self.provider.get_joint_angles(&self.metadata.joint_names))
                }
                "joint_angular_velocities" => futures.push(
                    self.provider
                        .get_joint_angular_velocities(&self.metadata.joint_names),
                ),
                "projected_gravity" => futures.push(self.provider.get_projected_gravity()),
                "accelerometer" => futures.push(self.provider.get_accelerometer()),
                "gyroscope" => futures.push(self.provider.get_gyroscope()),
                "command" => futures.push(self.provider.get_command()),
                "carry" => futures.push(self.provider.get_carry(carry.clone())),
                _ => return Err(format!("Unknown input name: {}", name).into()),
            }
        }

        let results = future::try_join_all(futures).await?;
        let mut inputs = HashMap::new();
        for (name, value) in input_names.iter().zip(results) {
            inputs.insert(name.clone(), value);
        }

        // Convert inputs to ONNX values
        let mut input_values: Vec<(&str, Value)> = Vec::new();
        for input in &self.step_session.inputs {
            let input_data = inputs
                .get(&input.name)
                .ok_or_else(|| format!("Missing input: {}", input.name))?;
            let input_value = Value::from_array(input_data.view())?.into_dyn();
            input_values.push((input.name.as_str(), input_value));
        }

        // Run the model
        let outputs = self.step_session.run(input_values)?;
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let carry_tensor = outputs[1].try_extract_tensor::<f32>()?;

        Ok((
            output_tensor.view().to_owned(),
            carry_tensor.view().to_owned(),
        ))
    }

    pub async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.provider
            .take_action(self.metadata.joint_names.clone(), action)
            .await?;
        Ok(())
    }

    pub async fn get_joint_angles(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let joint_names = &self.metadata.joint_names;
        let joint_angles = self.provider.get_joint_angles(joint_names).await?;
        Ok(joint_angles)
    }
}
