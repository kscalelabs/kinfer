use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use ndarray::{Array, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    joint_names: Vec<String>,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn get_joint_angles(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>>;
    async fn get_joint_angular_velocities(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>>;
    async fn get_projected_gravity(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>>;
    async fn get_accelerometer(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>>;
    async fn get_gyroscope(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>>;
    async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct ModelRunner {
    init_session: Session,
    step_session: Session,
    joint_names: Vec<String>,
    provider: Arc<dyn ModelProvider>,
}

impl ModelRunner {
    pub async fn new<P: AsRef<Path>>(
        model_path: P,
        input_provider: Arc<dyn ModelProvider>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(model_path).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;

        let mut archive = tar::Archive::new(std::io::Cursor::new(buffer));

        // Extract and validate joint names
        let mut joint_names_file = None;
        let mut init_fn = None;
        let mut step_fn = None;

        for entry in archive.entries()? {
            let entry = entry?;
            match entry.path()?.to_str() {
                Some("joint_names.json") => joint_names_file = Some(entry),
                Some("init_fn.onnx") => init_fn = Some(entry),
                Some("step_fn.onnx") => step_fn = Some(entry),
                _ => continue,
            }
        }

        let metadata: ModelMetadata =
            serde_json::from_reader(joint_names_file.ok_or("Missing joint_names.json")?)?;
        let init_session = Self::load_session(init_fn.ok_or("Missing init_fn.onnx")?)?;
        let step_session = Self::load_session(step_fn.ok_or("Missing step_fn.onnx")?)?;

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
        Self::validate_step_fn(&step_session, metadata.joint_names.len(), &carry_shape)?;

        Ok(Self {
            init_session,
            step_session,
            joint_names: metadata.joint_names,
            provider: input_provider,
        })
    }

    fn load_session<R: std::io::Read>(
        mut reader: R,
    ) -> Result<Session, Box<dyn std::error::Error>> {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        Ok(Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_memory(&buffer)?)
    }

    fn validate_step_fn(
        session: &Session,
        num_joints: usize,
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
        let mut inputs = HashMap::new();

        // Get all required inputs
        inputs.insert(
            "joint_angles".to_string(),
            self.provider.get_joint_angles(&self.joint_names).await?,
        );
        inputs.insert(
            "joint_angular_velocities".to_string(),
            self.provider
                .get_joint_angular_velocities(&self.joint_names)
                .await?,
        );
        inputs.insert(
            "projected_gravity".to_string(),
            self.provider.get_projected_gravity().await?,
        );
        inputs.insert(
            "accelerometer".to_string(),
            self.provider.get_accelerometer().await?,
        );
        inputs.insert(
            "gyroscope".to_string(),
            self.provider.get_gyroscope().await?,
        );
        inputs.insert("carry".to_string(), carry);

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
}
