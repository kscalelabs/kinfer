use crate::logger::StepLogger;
use crate::types::{InputType, ModelMetadata};
use async_trait::async_trait;
use chrono;
use flate2::read::GzDecoder;
use ndarray::{Array, IxDyn};
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use tar::Archive;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Provider error: {0}")]
    Provider(String),
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn get_inputs(
        &self,
        input_types: &[InputType],
        metadata: &ModelMetadata,
    ) -> Result<HashMap<InputType, Array<f32, IxDyn>>, ModelError>;

    async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError>;
}

pub struct ModelRunner {
    init_session: Session,
    step_session: Session,
    metadata: ModelMetadata,
    provider: Arc<dyn ModelProvider>,
    logger: Option<Arc<StepLogger>>,
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

        let logger = if let Ok(log_dir) = std::env::var("KINFER_LOG_PATH") {
            let log_dir_path = std::path::Path::new(&log_dir);

            // Create the directory if it doesn't exist
            if !log_dir_path.exists() {
                std::fs::create_dir_all(log_dir_path)?;
            }

            // Use uuid if found, otherwise timestamp
            let log_name = std::env::var("KINFER_LOG_UUID").unwrap_or_else(|_| {
                chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string()
            });

            let log_file_path = log_dir_path.join(format!("{}.ndjson", log_name));

            Some(StepLogger::new(log_file_path).map(Arc::new)?)
        } else {
            None
        };

        Ok(Self {
            init_session,
            step_session,
            metadata,
            provider: input_provider,
            logger,
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

            let input_type = InputType::from_name(&input.name)?;
            let expected_shape = input_type.get_shape(metadata);
            let expected_shape_i64: Vec<i64> = expected_shape.iter().map(|&x| x as i64).collect();
            if *dims != expected_shape_i64 {
                return Err(format!(
                    "Expected input shape {:?}, got {:?}",
                    expected_shape_i64, dims
                )
                .into());
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

    pub async fn get_inputs(
        &self,
        input_types: &[InputType],
    ) -> Result<HashMap<InputType, Array<f32, IxDyn>>, ModelError> {
        self.provider.get_inputs(input_types, &self.metadata).await
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
        let mut input_types = Vec::new();
        let mut inputs = HashMap::new();
        for name in &input_names {
            match name.as_str() {
                "joint_angles" => {
                    input_types.push(InputType::JointAngles);
                }
                "joint_angular_velocities" => {
                    input_types.push(InputType::JointAngularVelocities);
                }
                "initial_heading" => {
                    input_types.push(InputType::InitialHeading);
                }
                "quaternion" => {
                    input_types.push(InputType::Quaternion);
                }
                "projected_gravity" => {
                    input_types.push(InputType::ProjectedGravity);
                }
                "accelerometer" => {
                    input_types.push(InputType::Accelerometer);
                }
                "gyroscope" => {
                    input_types.push(InputType::Gyroscope);
                }
                "command" => {
                    input_types.push(InputType::Command);
                }
                "time" => {
                    input_types.push(InputType::Time);
                }
                "carry" => {
                    inputs.insert(InputType::Carry, carry.clone());
                }
                _ => return Err(format!("Unknown input name: {}", name).into()),
            }
        }

        // Gets the input values.
        let result = self
            .provider
            .get_inputs(&input_types, &self.metadata)
            .await?;

        // Adds the input values to the input map.
        inputs.extend(result);

        // Convert inputs to ONNX values
        let mut input_values: Vec<(&str, Value)> = Vec::new();
        for input in &self.step_session.inputs {
            let input_type = InputType::from_name(&input.name)?;
            let input_data = inputs
                .get(&input_type)
                .ok_or_else(|| format!("Missing input: {}", input.name))?;
            let input_value = Value::from_array(input_data.view())?.into_dyn();
            input_values.push((input.name.as_str(), input_value));
        }

        // Run the model
        let outputs = self.step_session.run(input_values)?;
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let carry_tensor = outputs[1].try_extract_tensor::<f32>()?;

        // Log the step if needed
        if let Some(lg) = &self.logger {
            let joint_angles_opt = inputs
                .get(&InputType::JointAngles)
                .map(|a| a.as_slice().unwrap());
            let joint_vels_opt = inputs
                .get(&InputType::JointAngularVelocities)
                .map(|a| a.as_slice().unwrap());
            let initial_heading_opt = inputs
                .get(&InputType::InitialHeading)
                .map(|a| a.as_slice().unwrap());
            let quaternion_opt = inputs
                .get(&InputType::Quaternion)
                .map(|a| a.as_slice().unwrap());
            let projected_g_opt = inputs
                .get(&InputType::ProjectedGravity)
                .map(|a| a.as_slice().unwrap());
            let accel_opt = inputs
                .get(&InputType::Accelerometer)
                .map(|a| a.as_slice().unwrap());
            let gyro_opt = inputs
                .get(&InputType::Gyroscope)
                .map(|a| a.as_slice().unwrap());
            let command_opt = inputs
                .get(&InputType::Command)
                .map(|a| a.as_slice().unwrap());
            let output_opt = Some(output_tensor.as_slice().unwrap());

            lg.log_step(
                joint_angles_opt,
                joint_vels_opt,
                initial_heading_opt,
                quaternion_opt,
                projected_g_opt,
                accel_opt,
                gyro_opt,
                command_opt,
                output_opt,
            );
        }

        Ok((
            output_tensor.view().to_owned(),
            carry_tensor.view().to_owned(),
        ))
    }

    pub async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.provider.take_action(action, &self.metadata).await?;
        Ok(())
    }
}
