use crate::logger::StepLogger;
use crate::types::{InputType, ModelMetadata};
use chrono;
use flate2::read::GzDecoder;
use ndarray::{Array, IxDyn};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{Tensor, TensorValueType, Value, ValueRef},
};
use std::fs::File;
use std::hint::spin_loop;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tar::Archive;

#[inline]
fn wait_until(deadline: Instant, spin: Duration) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remain = deadline - now;

        // Sleep away the bulk…
        if remain > spin + Duration::from_micros(150) {
            std::thread::sleep(remain - spin);
            continue;
        }
        // Short yield if we're close
        if remain > spin {
            std::thread::yield_now();
            continue;
        }
        // Final busy wait for sub-millisecond precision
        while Instant::now() < deadline {
            spin_loop();
        }
        break;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Provider error: {0}")]
    Provider(String),
}

pub trait ModelProvider: Send + Sync {
    fn pre_fetch_inputs(
        &self,
        input_buffers: &[(InputType, Tensor<f32>)],
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError>;

    fn get_inputs(
        &self,
        input_buffers: &mut [(InputType, Tensor<f32>)],
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError>;

    fn take_action(
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
    pre_fetch_time: Option<Duration>,
    inputs_buffer: Vec<(InputType, Tensor<f32>)>,
    #[allow(dead_code)] // Used implicitly by tensors in inputs_buffer
    allocator: Allocator,
    logger: Option<Arc<StepLogger>>,
}

impl ModelRunner {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        input_provider: Arc<dyn ModelProvider>,
        pre_fetch_time: Option<Duration>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(model_path)?;

        // Read entire file into memory
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

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

        // Pre-allocate buffers based on model metadata
        let input_names: Vec<String> = step_session.inputs.iter().map(|i| i.name.clone()).collect();
        let mut inputs_buffer: Vec<(InputType, Tensor<f32>)> =
            Vec::with_capacity(input_names.len());

        // Can update this allocator to use CUDA later on if necessary.
        let allocator = Allocator::new(
            &step_session,
            MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::CPUInput,
            )?,
        )?;

        // Pre-allocate arrays for each input type
        for name in &input_names {
            match name.as_str() {
                "joint_angles" => {
                    let shape = InputType::JointAngles.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::JointAngles,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "joint_angular_velocities" => {
                    let shape = InputType::JointAngularVelocities.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::JointAngularVelocities,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "projected_gravity" => {
                    let shape = InputType::ProjectedGravity.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::ProjectedGravity,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "accelerometer" => {
                    let shape = InputType::Accelerometer.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::Accelerometer,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "gyroscope" => {
                    let shape = InputType::Gyroscope.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::Gyroscope,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "command" => {
                    let shape = InputType::Command.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::Command,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "time" => {
                    let shape = InputType::Time.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::Time,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                "carry" => {
                    let shape = InputType::Carry.get_shape(&metadata);
                    inputs_buffer.push((
                        InputType::Carry,
                        Tensor::<f32>::new(&allocator, shape.as_slice())?,
                    ));
                }
                _ => return Err(format!("Unknown input name: {name}").into()),
            }
        }

        let logger = if let Ok(log_dir) = std::env::var("KINFER_LOG_PATH") {
            let log_dir_path = std::path::Path::new(&log_dir);

            // Create the directory if it doesn't exist
            if !log_dir_path.exists() {
                std::fs::create_dir_all(log_dir_path)?;
            }

            // Use uuid if found, otherwise timestamp
            let log_name = std::env::var("KINFER_LOG_UUID")
                .unwrap_or_else(|_| chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string());

            let log_file_path = log_dir_path.join(format!("{log_name}.ndjson"));

            Some(StepLogger::new(log_file_path).map(Arc::new)?)
        } else {
            None
        };

        Ok(Self {
            init_session,
            step_session,
            metadata,
            provider: input_provider,
            pre_fetch_time,
            inputs_buffer,
            allocator,
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
                return Err(
                    format!("Expected input shape {expected_shape_i64:?}, got {dims:?}").into(),
                );
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
            return Err(
                format!("Expected output shape [{num_joints}], got {output_shape:?}").into(),
            );
        }

        let infered_carry_shape = session.outputs[1]
            .output_type
            .tensor_dimensions()
            .ok_or("Missing tensor type")?;
        if *infered_carry_shape != *carry_shape {
            return Err(format!(
                "Expected carry shape {carry_shape:?}, got {infered_carry_shape:?}"
            )
            .into());
        }

        Ok(())
    }

    pub fn pre_fetch_inputs(&self) -> Result<(), ModelError> {
        self.provider
            .pre_fetch_inputs(&self.inputs_buffer, &self.metadata)
    }

    fn get_inputs(&mut self) -> Result<(), ModelError> {
        self.provider
            .get_inputs(&mut self.inputs_buffer, &self.metadata)
    }

    fn get_input_buffer_mut(&mut self, input_type: InputType) -> Option<&mut Tensor<f32>> {
        self.inputs_buffer
            .iter_mut()
            .find(|(t, _)| *t == input_type)
            .map(|(_, v)| v)
    }

    fn get_input_buffer_values(
        &self,
        input_type: InputType,
    ) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
        if let Some(pair) = self.inputs_buffer.iter().find(|(t, _)| *t == input_type) {
            let arr = pair
                .1
                .try_extract_tensor::<f32>()?
                .clone()
                .view()
                .to_owned()
                .as_slice()
                .map(|x| x.to_vec());

            Ok(arr)
        } else {
            Ok(None)
        }
    }

    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let output_data = {
            let input_values: Vec<(&str, Value)> = Vec::new();
            let output_values = self.init_session.run(input_values)?;
            if output_values.len() != 1 {
                return Err("Expected exactly one output value".into());
            }
            let output_tensor = output_values[0].try_extract_tensor::<f32>()?;
            output_tensor.view().to_owned()
        };

        // Copies the output tensor to the carry buffer.
        if let Some(carry_val) = self.get_input_buffer_mut(InputType::Carry) {
            let mut carry_view = carry_val.try_extract_tensor_mut()?;
            carry_view.assign(&output_data);
        }

        Ok(())
    }

    pub fn step(&mut self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        // Pre-fetches the inputs if requested, then sleep for a short amount of time.
        if let Some(pre_fetch_time) = self.pre_fetch_time {
            let start_time = std::time::Instant::now();
            self.pre_fetch_inputs()?;
            let elapsed = start_time.elapsed();
            if elapsed < pre_fetch_time {
                std::thread::sleep(pre_fetch_time - elapsed);
            }
        }

        // Gets the input values.
        self.get_inputs()?;

        // Run the model
        let (output, carry) = {
            let inputs: Vec<(&str, ValueRef<'_, TensorValueType<f32>>)> = self
                .inputs_buffer
                .iter()
                .map(|(t, v)| (t.get_name(), v.view()))
                .collect();

            // `inputs` is moved into `run`, and then dropped at block end
            let outputs = self.step_session.run(inputs)?;
            if outputs.len() != 2 {
                return Err("Expected exactly two outputs".into());
            }

            let output = outputs[0].try_extract_tensor::<f32>()?.view().to_owned();
            let carry = outputs[1].try_extract_tensor::<f32>()?.view().to_owned();

            (output, carry)
        };

        // Populates the carry buffer.
        if let Some(carry_val) = self.get_input_buffer_mut(InputType::Carry) {
            let mut carry_view = carry_val.try_extract_tensor_mut()?;
            carry_view.assign(&carry);
        }

        if let Some(logger) = &self.logger {
            let joint_angles = self.get_input_buffer_values(InputType::JointAngles)?;
            let joint_vels = self.get_input_buffer_values(InputType::JointAngularVelocities)?;
            let projected_g = self.get_input_buffer_values(InputType::ProjectedGravity)?;
            let accel = self.get_input_buffer_values(InputType::Accelerometer)?;
            let gyro = self.get_input_buffer_values(InputType::Gyroscope)?;
            let command = self.get_input_buffer_values(InputType::Command)?;
            let out = output.as_slice().map(|x| x.to_vec());

            logger.log_step(
                joint_angles,
                joint_vels,
                projected_g,
                accel,
                gyro,
                command,
                out,
            )
        }

        Ok(output)
    }

    pub fn take_action(&self, action: Array<f32, IxDyn>) -> Result<(), Box<dyn std::error::Error>> {
        self.provider.take_action(action, &self.metadata)?;
        Ok(())
    }

    pub fn get_joint_count(&self) -> usize {
        self.metadata.joint_names.len()
    }

    pub fn step_and_take_action(&mut self) -> Result<(), ModelError> {
        let output = self
            .step()
            .map_err(|e| ModelError::Provider(e.to_string()))?;

        self.take_action(output)
            .map_err(|e| ModelError::Provider(e.to_string()))?;

        Ok(())
    }

    pub fn run(
        &mut self,
        dt: Duration,
        total_runtime: Option<Duration>,
        total_steps: Option<u64>,
    ) -> Result<(), ModelError> {
        self.init()
            .map_err(|e| ModelError::Provider(e.to_string()))?;

        let start = Instant::now();
        let mut deadline = start + dt;
        let mut steps = 0u64;

        const SPIN: Duration = Duration::from_micros(300); // tune 100–500µs

        loop {
            wait_until(deadline, SPIN);
            let tick_start = Instant::now();

            // Execute the step
            self.step_and_take_action()?;

            // Termination checks with fresh time
            if let Some(rt) = total_runtime {
                if tick_start.duration_since(start) >= rt {
                    break;
                }
            }
            if let Some(max_steps) = total_steps {
                steps += 1;
                if steps >= max_steps {
                    break;
                }
            }

            // Advance absolute schedule; catch up if we overran
            deadline += dt;
            let now = Instant::now();
            if deadline + dt < now {
                // If we fell far behind, skip ahead to avoid spiral-of-death
                let behind = now.duration_since(deadline);
                let skip = (behind.as_nanos() / dt.as_nanos()) as u32;
                deadline += dt * (skip + 1);
            }
        }

        Ok(())
    }
}
