use async_trait::async_trait;
use kinfer::model::{ModelError, ModelProvider, ModelRunner};
use kinfer::runtime::ModelRuntime;
use kinfer::types::{InputType, ModelMetadata};
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

#[pyclass]
#[gen_stub_pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PyModelMetadata {
    #[pyo3(get, set)]
    pub joint_names: Vec<String>,
    #[pyo3(get, set)]
    pub num_commands: Option<usize>,
    #[pyo3(get, set)]
    pub carry_size: Vec<usize>,
}

#[pymethods]
#[gen_stub_pymethods]
impl PyModelMetadata {
    #[new]
    fn __new__(
        joint_names: Vec<String>,
        num_commands: Option<usize>,
        carry_size: Vec<usize>,
    ) -> Self {
        Self {
            joint_names,
            num_commands,
            carry_size,
        }
    }

    fn to_json(&self) -> PyResult<String> {
        let metadata = ModelMetadata {
            joint_names: self.joint_names.clone(),
            num_commands: self.num_commands,
            carry_size: self.carry_size.clone(),
        }
        .to_json()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(metadata)
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

#[pyfunction]
#[gen_stub_pyfunction]
fn metadata_from_json(json: &str) -> PyResult<PyModelMetadata> {
    let metadata = ModelMetadata::model_validate_json(json.to_string()).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid model metadata: {}", e))
    })?;
    Ok(PyModelMetadata::from(metadata))
}

impl From<ModelMetadata> for PyModelMetadata {
    fn from(metadata: ModelMetadata) -> Self {
        Self {
            joint_names: metadata.joint_names,
            num_commands: metadata.num_commands,
            carry_size: metadata.carry_size,
        }
    }
}

impl From<&ModelMetadata> for PyModelMetadata {
    fn from(metadata: &ModelMetadata) -> Self {
        Self {
            joint_names: metadata.joint_names.clone(),
            num_commands: metadata.num_commands,
            carry_size: metadata.carry_size.clone(),
        }
    }
}

impl From<PyModelMetadata> for ModelMetadata {
    fn from(metadata: PyModelMetadata) -> Self {
        Self {
            joint_names: metadata.joint_names,
            num_commands: metadata.num_commands,
            carry_size: metadata.carry_size,
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
    m.add_function(wrap_pyfunction!(metadata_from_json, m)?)?;
    m.add_class::<ModelProviderABC>()?;
    m.add_class::<PyModelRunner>()?;
    m.add_class::<PyModelRuntime>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);