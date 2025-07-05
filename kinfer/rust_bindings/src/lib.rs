use async_trait::async_trait;
use kinfer::model::{ModelError, ModelProvider, ModelRunner};
use kinfer::runtime::ModelRuntime;
use kinfer::types::{InputType, ModelMetadata, JointBias, CommandTypeInfo, CommandField};
use ndarray::{Array, Ix1, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::Mutex;

type StepResult = (Py<PyArrayDyn<f32>>, Py<PyArrayDyn<f32>>);

// Custom error type for Send/Sync compatibility
#[derive(Debug)]
struct SendError(String);

unsafe impl Send for SendError {}
unsafe impl Sync for SendError {}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[pyfunction]
#[gen_stub_pyfunction]
fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pyclass]
#[gen_stub_pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PyInputType {
    pub input_type: InputType,
}

impl From<InputType> for PyInputType {
    fn from(input_type: InputType) -> Self {
        Self { input_type }
    }
}

impl From<PyInputType> for InputType {
    fn from(input_type: PyInputType) -> Self {
        input_type.input_type
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInputType {
    #[new]
    fn __new__(input_type: &str) -> PyResult<Self> {
        let input_type = InputType::from_name(input_type).map_or_else(
            |_| {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid input type: {} (must be one of {})",
                    input_type,
                    InputType::get_names().join(", "),
                )))
            },
            Ok,
        )?;
        Ok(Self { input_type })
    }

    fn get_name(&self) -> String {
        self.input_type.get_name().to_string()
    }

    fn get_shape(&self, metadata: PyModelMetadata) -> Vec<usize> {
        self.input_type.get_shape(&metadata.into())
    }

    fn __repr__(&self) -> String {
        format!("InputType({})", self.get_name())
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyInputType>() {
            Ok(self == &other)
        } else {
            Ok(false)
        }
    }
}



#[pyfunction]
#[gen_stub_pyfunction]
fn metadata_from_json(json: &str) -> PyResult<PyModelMetadata> {
    let metadata = ModelMetadata::model_validate_json(json.to_string()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid model metadata: {}", e))
    })?;
    Ok(PyModelMetadata::from(metadata))
}

impl From<&ModelMetadata> for PyModelMetadata {
    fn from(metadata: &ModelMetadata) -> Self {
        let joint_biases = metadata.joint_biases.as_ref().map(|biases| {
            biases.iter().map(|bias| PyJointBias {
                joint_name: bias.joint_name.clone(),
                reference_angle: bias.reference_angle,
                weight: bias.weight,
            }).collect()
        });

        let command_type_info = metadata.command_type_info.as_ref().map(|info| {
            PyCommandTypeInfo {
                command_type: info.command_type.clone(),
                description: info.description.clone(),
                fields: info.fields.iter().map(|field| PyCommandField {
                    name: field.name.clone(),
                    description: field.description.clone(),
                    units: field.units.clone(),
                    range: field.range,
                }).collect(),
            }
        });

        Self {
            joint_names: metadata.joint_names.clone(),
            num_commands: metadata.num_commands,
            carry_size: metadata.carry_size.clone(),
            joint_biases,
            command_type_info,
            kinfer_version: metadata.kinfer_version.clone(),
            training_metadata: None, // Cannot convert easily
        }
    }
}

impl From<PyModelMetadata> for ModelMetadata {
    fn from(metadata: PyModelMetadata) -> Self {
        let joint_biases = metadata.joint_biases.map(|biases| {
            biases.into_iter().map(|bias| JointBias {
                joint_name: bias.joint_name,
                reference_angle: bias.reference_angle,
                weight: bias.weight,
            }).collect()
        });

        let command_type_info = metadata.command_type_info.map(|info| {
            CommandTypeInfo {
                command_type: info.command_type,
                description: info.description,
                fields: info.fields.into_iter().map(|field| CommandField {
                    name: field.name,
                    description: field.description,
                    units: field.units,
                    range: field.range,
                }).collect(),
            }
        });

        Self {
            joint_names: metadata.joint_names,
            num_commands: metadata.num_commands,
            carry_size: metadata.carry_size,
            joint_biases,
            command_type_info,
            kinfer_version: metadata.kinfer_version,
            training_metadata: None, // Cannot convert PyAny easily
        }
    }
}

impl From<kinfer::types::ModelMetadata> for PyModelMetadata {
    fn from(metadata: kinfer::types::ModelMetadata) -> Self {
        let joint_biases = metadata.joint_biases.map(|biases| {
            biases.into_iter().map(|bias| PyJointBias {
                joint_name: bias.joint_name,
                reference_angle: bias.reference_angle,
                weight: bias.weight,
            }).collect()
        });

        let command_type_info = metadata.command_type_info.map(|info| {
            PyCommandTypeInfo {
                command_type: info.command_type,
                description: info.description,
                fields: info.fields.into_iter().map(|field| PyCommandField {
                    name: field.name,
                    description: field.description,
                    units: field.units,
                    range: field.range,
                }).collect(),
            }
        });

        Self {
            joint_names: metadata.joint_names,
            num_commands: metadata.num_commands,
            carry_size: metadata.carry_size,
            joint_biases,
            command_type_info,
            kinfer_version: metadata.kinfer_version,
            training_metadata: None, // TODO: Implement proper conversion
        }
    }
}

#[pyclass]
#[gen_stub_pyclass]
#[derive(Debug, Clone, PartialEq)]
struct PyJointBias {
    #[pyo3(get, set)]
    pub joint_name: String,
    #[pyo3(get, set)]
    pub reference_angle: f64,
    #[pyo3(get, set)]
    pub weight: f64,
}

#[pymethods]
#[gen_stub_pymethods]
impl PyJointBias {
    #[new]
    fn __new__(joint_name: String, reference_angle: f64, weight: f64) -> Self {
        Self {
            joint_name,
            reference_angle,
            weight,
        }
    }
    
    fn __repr__(&self) -> String {
        format!("JointBias(joint_name='{}', reference_angle={}, weight={})", 
                self.joint_name, self.reference_angle, self.weight)
    }
}

#[pyclass]
#[gen_stub_pyclass]
#[derive(Debug, Clone, PartialEq)]
struct PyCommandField {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub units: Option<String>,
    #[pyo3(get, set)]
    pub range: Option<(f64, f64)>,
}

#[pymethods]
#[gen_stub_pymethods]
impl PyCommandField {
    #[new]
    fn __new__(name: String, description: String, units: Option<String>, range: Option<(f64, f64)>) -> Self {
        Self {
            name,
            description,
            units,
            range,
        }
    }
    
    fn __repr__(&self) -> String {
        format!("CommandField(name='{}', description='{}', units={:?}, range={:?})", 
                self.name, self.description, self.units, self.range)
    }
}

#[pyclass]
#[gen_stub_pyclass]
#[derive(Debug, Clone, PartialEq)]
struct PyCommandTypeInfo {
    #[pyo3(get, set)]
    pub command_type: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub fields: Vec<PyCommandField>,
}

#[pymethods]
#[gen_stub_pymethods]
impl PyCommandTypeInfo {
    #[new]
    fn __new__(command_type: String, description: String, fields: Vec<PyCommandField>) -> Self {
        Self {
            command_type,
            description,
            fields,
        }
    }
    
    fn __repr__(&self) -> String {
        format!("CommandTypeInfo(command_type='{}', description='{}', fields={:?})", 
                self.command_type, self.description, self.fields)
    }
}

#[pyclass]
#[gen_stub_pyclass]
#[derive(Debug)]
struct PyModelMetadata {
    #[pyo3(get, set)]
    pub joint_names: Vec<String>,
    #[pyo3(get, set)]
    pub num_commands: Option<usize>,
    #[pyo3(get, set)]
    pub carry_size: Vec<usize>,
    #[pyo3(get, set)]
    pub joint_biases: Option<Vec<PyJointBias>>,
    #[pyo3(get, set)]
    pub command_type_info: Option<PyCommandTypeInfo>,
    #[pyo3(get, set)]
    pub kinfer_version: Option<String>,
    #[pyo3(get, set)]
    pub training_metadata: Option<HashMap<String, Py<PyAny>>>,
}

impl Clone for PyModelMetadata {
    fn clone(&self) -> Self {
        Self {
            joint_names: self.joint_names.clone(),
            num_commands: self.num_commands,
            carry_size: self.carry_size.clone(),
            joint_biases: self.joint_biases.clone(),
            command_type_info: self.command_type_info.clone(),
            kinfer_version: self.kinfer_version.clone(),
            training_metadata: None, // Skip cloning PyAny
        }
    }
}

impl PartialEq for PyModelMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.joint_names == other.joint_names
            && self.num_commands == other.num_commands
            && self.carry_size == other.carry_size
            && self.joint_biases == other.joint_biases
            && self.command_type_info == other.command_type_info
            && self.kinfer_version == other.kinfer_version
            // Skip comparing training_metadata
    }
}
impl Eq for PyModelMetadata {}

impl PyModelMetadata {
    //fn to_rust_metadata(&self) -> PyResult<kinfer::types::ModelMetadata> {
    fn to_rust_metadata(&self) -> PyResult<ModelMetadata> {
        // Convert joint biases
        let joint_biases = self.joint_biases.as_ref().map(|biases| {
            biases.iter().map(|bias| kinfer::types::JointBias {
                joint_name: bias.joint_name.clone(),
                reference_angle: bias.reference_angle,
                weight: bias.weight,
            }).collect()
        });

        // Convert command type info
        let command_type_info = self.command_type_info.as_ref().map(|info| {
            kinfer::types::CommandTypeInfo {
                command_type: info.command_type.clone(),
                description: info.description.clone(),
                fields: info.fields.iter().map(|field| kinfer::types::CommandField {
                    name: field.name.clone(),
                    description: field.description.clone(),
                    units: field.units.clone(),
                    range: field.range,
                }).collect(),
            }
        });

        // Convert training metadata (simplified - would need more complex handling for PyAny)
        let training_metadata = None; // TODO: Implement proper PyAny to serde_json::Value conversion

        Ok(kinfer::types::ModelMetadata {
            joint_names: self.joint_names.clone(),
            num_commands: self.num_commands,
            carry_size: self.carry_size.clone(),
            joint_biases,
            command_type_info,
            kinfer_version: self.kinfer_version.clone(),
            training_metadata,
        })
    }
}

#[pymethods]
#[gen_stub_pymethods]
impl PyModelMetadata {
    #[new]
    fn __new__(
        joint_names: Vec<String>,
        num_commands: Option<usize>,
        carry_size: Vec<usize>,
        joint_biases: Option<Vec<PyJointBias>>,
        command_type_info: Option<PyCommandTypeInfo>,
        kinfer_version: Option<String>,
        training_metadata: Option<HashMap<String, Py<PyAny>>>,
    ) -> Self {
        Self {
            joint_names,
            num_commands,
            carry_size,
            joint_biases,
            command_type_info,
            kinfer_version,
            training_metadata,
        }
    }

    fn to_json(&self) -> PyResult<String> {
        // Convert to Rust types for serialization
        let rust_metadata = self.to_rust_metadata()?;
        rust_metadata.to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn get_joint_bias_by_name(&self, joint_name: &str) -> Option<PyJointBias> {
        self.joint_biases.as_ref()?.iter().find(|bias| bias.joint_name == joint_name).cloned()
    }

    fn get_command_type(&self) -> Option<String> {
        self.command_type_info.as_ref().map(|info| info.command_type.clone())
    }

    fn get_command_description(&self) -> Option<String> {
        self.command_type_info.as_ref().map(|info| info.description.clone())
    }

    fn validate_command_compatibility(&self, expected_command_type: &str) -> PyResult<()> {
        let rust_metadata = self.to_rust_metadata()?;
        rust_metadata.validate_command_compatibility(expected_command_type)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
    
    
    
    fn __repr__(&self) -> PyResult<String> {
        let json = self.to_json()?;
        Ok(format!("ModelMetadata({:?})", json))
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyModelMetadata>() {
            Ok(self == &other)
        } else {
            Ok(false)
        }
    }

}

#[pyclass(subclass)]
#[gen_stub_pyclass]
struct ModelProviderABC;

#[gen_stub_pymethods]
#[pymethods]
impl ModelProviderABC {
    #[new]
    fn __new__() -> Self {
        ModelProviderABC
    }

    fn get_inputs<'py>(
        &self,
        input_types: Vec<String>,
        metadata: PyModelMetadata,
    ) -> PyResult<HashMap<String, Bound<'py, PyArrayDyn<f32>>>> {
        Err(PyNotImplementedError::new_err(format!(
            "Must override get_inputs with {} input types {:?} and metadata {:?}",
            input_types.len(),
            input_types,
            metadata
        )))
    }

    fn take_action(
        &self,
        action: Bound<'_, PyArray1<f32>>,
        metadata: PyModelMetadata,
    ) -> PyResult<()> {
        let n = action.len()?;
        if metadata.joint_names.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} joints, got {} action elements",
                metadata.joint_names.len(),
                n
            )));
        }
        Err(PyNotImplementedError::new_err(format!(
            "Must override take_action with {} action elements",
            n
        )))
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct PyModelProvider {
    obj: Arc<Py<ModelProviderABC>>,
}

#[pymethods]
impl PyModelProvider {
    #[new]
    fn __new__(obj: Py<ModelProviderABC>) -> Self {
        Self { obj: Arc::new(obj) }
    }
}

#[async_trait]
impl ModelProvider for PyModelProvider {
    async fn get_inputs(
        &self,
        input_types: &[InputType],
        metadata: &ModelMetadata,
    ) -> Result<HashMap<InputType, Array<f32, IxDyn>>, ModelError> {
        let input_names: Vec<String> = input_types
            .iter()
            .map(|t| t.get_name().to_string())
            .collect();
        let result = Python::with_gil(|py| -> PyResult<HashMap<InputType, Array<f32, IxDyn>>> {
            let obj = self.obj.clone();
            let args = (input_names.clone(), PyModelMetadata::from(metadata.clone()));
            let result = obj.call_method(py, "get_inputs", args, None)?;
            let dict: HashMap<String, Vec<f32>> = result.extract(py)?;
            let mut arrays = HashMap::new();
            for (i, name) in input_names.iter().enumerate() {
                let array = dict.get(name).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Missing input: {}",
                        name
                    ))
                })?;
                arrays.insert(input_types[i], Array::from_vec(array.clone()).into_dyn());
            }
            Ok(arrays)
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(result)
    }

    async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError> {
        Python::with_gil(|py| -> PyResult<()> {
            let obj = self.obj.clone();
            let action_1d = action
                .into_dimensionality::<Ix1>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let args = (
                PyArray1::from_array(py, &action_1d),
                PyModelMetadata::from(metadata.clone()),
            );
            obj.call_method(py, "take_action", args, None)?;
            Ok(())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(())
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct PyModelRunner {
    runner: Arc<ModelRunner>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelRunner {
    #[new]
    fn __new__(model_path: String, provider: Py<ModelProviderABC>) -> PyResult<Self> {
        let input_provider = Arc::new(PyModelProvider::__new__(provider));

        // Create a single runtime to be reused for all operations
        let runtime = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?);

        let runner = runtime.block_on(async {
            ModelRunner::new(model_path, input_provider)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })?;

        Ok(Self {
            runner: Arc::new(runner),
            runtime,
        })
    }

    // Reuse runtime and release GIL
    fn init(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        let runner = self.runner.clone();
        let runtime = self.runtime.clone();

        let result = Python::with_gil(|py| {
            // Release GIL during async operation
            py.allow_threads(|| {
                runtime.block_on(async {
                    runner
                        .init()
                        .await
                        .map_err(|e| SendError(e.to_string()))
                })
            })
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.0))?;

        Python::with_gil(|py| {
            let array = numpy::PyArray::from_array(py, &result);
            Ok(array.into())
        })
    }

    // Reuse runtime and release GIL
    fn step(&self, carry: Py<PyArrayDyn<f32>>) -> PyResult<StepResult> {
        let runner = self.runner.clone();
        let runtime = self.runtime.clone();
        
        // Extract the carry array from Python with GIL
        let carry_array = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let carry_array = carry.bind(py);
            Ok(carry_array.to_owned_array())
        })?;

        // Release GIL during computation
        let result = Python::with_gil(|py| {
            py.allow_threads(|| {
                runtime.block_on(async {
                    runner
                        .step(carry_array)
                        .await
                        .map_err(|e| SendError(e.to_string()))
                })
            })
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.0))?;

        // Reacquire the GIL to convert results back to Python objects
        Python::with_gil(|py| {
            let (output, carry) = result;
            let output_array = numpy::PyArray::from_array(py, &output);
            let carry_array = numpy::PyArray::from_array(py, &carry);
            Ok((output_array.into(), carry_array.into()))
        })
    }

    // Reuse runtime and release GIL
    fn take_action(&self, action: Py<PyArrayDyn<f32>>) -> PyResult<()> {
        let runner = self.runner.clone();
        let runtime = self.runtime.clone();

        // Extract action data with GIL
        let action_array = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let action_array = action.bind(py);
            Ok(action_array.to_owned_array())
        })?;
        
        // Release GIL during computation
        Python::with_gil(|py| {
            py.allow_threads(|| {
                runtime.block_on(async {
                    runner
                        .take_action(action_array)
                        .await
                        .map_err(|e| SendError(e.to_string()))
                })
            })
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.0))?;

        Ok(())
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct PyModelRuntime {
    runtime: Arc<Mutex<ModelRuntime>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelRuntime {
    #[new]
    fn __new__(model_runner: PyModelRunner, dt: u64) -> PyResult<Self> {
        Ok(Self {
            runtime: Arc::new(Mutex::new(ModelRuntime::new(model_runner.runner, dt))),
        })
    }

    fn set_slowdown_factor(&self, slowdown_factor: i32) -> PyResult<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        runtime.set_slowdown_factor(slowdown_factor);
        Ok(())
    }

    fn set_magnitude_factor(&self, magnitude_factor: f32) -> PyResult<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        runtime.set_magnitude_factor(magnitude_factor);
        Ok(())
    }

    fn start(&self) -> PyResult<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        runtime
            .start()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn stop(&self) -> PyResult<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        runtime.stop();
        Ok(())
    }
}

#[pymodule]
fn rust_bindings(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_class::<PyInputType>()?;
    m.add_class::<PyModelMetadata>()?;
    m.add_class::<PyJointBias>()?;
    m.add_class::<PyCommandField>()?;
    m.add_class::<PyCommandTypeInfo>()?;
    m.add_function(wrap_pyfunction!(metadata_from_json, m)?)?;
    m.add_class::<ModelProviderABC>()?;
    m.add_class::<PyModelRunner>()?;
    m.add_class::<PyModelRuntime>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);