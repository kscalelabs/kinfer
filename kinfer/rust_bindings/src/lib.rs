use kinfer::model::{ModelError, ModelProvider, ModelRunner};
use kinfer::types::{InputType, ModelMetadata};
use ndarray::{Array, Ix1, IxDyn};
use numpy::{PyArray1, PyArrayDyn};
use ort::value::Tensor;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// Custom error type for Send/Sync compatibility
#[allow(dead_code)]
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
    pub command_names: Vec<String>,
    #[pyo3(get, set)]
    pub carry_size: Vec<usize>,
}

#[pymethods]
#[gen_stub_pymethods]
impl PyModelMetadata {
    #[new]
    fn __new__(
        joint_names: Vec<String>,
        command_names: Vec<String>,
        carry_size: Vec<usize>,
    ) -> Self {
        Self {
            joint_names,
            command_names,
            carry_size,
        }
    }

    fn to_json(&self) -> PyResult<String> {
        let metadata = ModelMetadata {
            joint_names: self.joint_names.clone(),
            command_names: self.command_names.clone(),
            carry_size: self.carry_size.clone(),
        }
        .to_json()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(metadata)
    }

    fn __repr__(&self) -> PyResult<String> {
        let json = self.to_json()?;
        Ok(format!("ModelMetadata({json:?})"))
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
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid model metadata: {e}"))
    })?;
    Ok(PyModelMetadata::from(metadata))
}

impl From<ModelMetadata> for PyModelMetadata {
    fn from(metadata: ModelMetadata) -> Self {
        Self {
            joint_names: metadata.joint_names,
            command_names: metadata.command_names,
            carry_size: metadata.carry_size,
        }
    }
}

impl From<&ModelMetadata> for PyModelMetadata {
    fn from(metadata: &ModelMetadata) -> Self {
        Self {
            joint_names: metadata.joint_names.clone(),
            command_names: metadata.command_names.clone(),
            carry_size: metadata.carry_size.clone(),
        }
    }
}

impl From<PyModelMetadata> for ModelMetadata {
    fn from(metadata: PyModelMetadata) -> Self {
        Self {
            joint_names: metadata.joint_names,
            command_names: metadata.command_names,
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

    fn pre_fetch_inputs(
        &self,
        input_types: Vec<String>,
        metadata: PyModelMetadata,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(format!(
            "Must override pre_fetch_inputs with {} input types {:?} and metadata {:?}",
            input_types.len(),
            input_types,
            metadata
        )))
    }

    fn get_inputs(
        &self,
        input_types: Vec<String>,
        metadata: PyModelMetadata,
    ) -> PyResult<HashMap<String, Py<PyArrayDyn<f32>>>> {
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
            "Must override take_action with {n} action elements"
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

impl ModelProvider for PyModelProvider {
    fn pre_fetch_inputs(
        &self,
        input_buffers: &[(InputType, Tensor<f32>)],
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError> {
        let input_names: Vec<String> = input_buffers
            .iter()
            .filter(|t| t.0 != InputType::Carry)
            .map(|t| t.0.get_name().to_string())
            .collect();

        Python::attach(|py| -> PyResult<()> {
            let obj = self.obj.clone();
            let args = (input_names.clone(), PyModelMetadata::from(metadata.clone()));
            obj.call_method(py, "pre_fetch_inputs", args, None)?;
            Ok(())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;

        Ok(())
    }

    fn get_inputs(
        &self,
        input_buffers: &mut [(InputType, Tensor<f32>)],
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError> {
        let input_names: Vec<String> = input_buffers
            .iter()
            .filter(|t| t.0 != InputType::Carry)
            .map(|t| t.0.get_name().to_string())
            .collect();

        Python::attach(|py| -> Result<(), Box<dyn std::error::Error>> {
            let obj = self.obj.clone();
            let args = (input_names.clone(), PyModelMetadata::from(metadata.clone()));
            let result = obj.call_method(py, "get_inputs", args, None)?;
            let dict: HashMap<String, Vec<f32>> = result.extract(py)?;

            for (name, array) in input_buffers.iter_mut().filter(|t| t.0 != InputType::Carry) {
                let name_str = name.get_name();
                let src = dict.get(name_str).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Missing input: {name_str}"
                    ))
                })?;

                // Check that the shape is correct, then copy the contents.
                let (shape, dst) = array.extract_raw_tensor_mut();
                let expected = shape.iter().try_fold(1usize, |acc, &d| {
                    usize::try_from(d).map(|dd| acc.checked_mul(dd).unwrap())
                })?;

                if expected != src.len() {
                    return Err(Box::<dyn std::error::Error>::from(format!(
                        "Shape mismatch for {name_str}: expected {expected} f32s, got {}",
                        src.len()
                    )));
                }

                dst.copy_from_slice(src);
            }
            Ok(())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;

        Ok(())
    }

    fn take_action(
        &self,
        action: Array<f32, IxDyn>,
        metadata: &ModelMetadata,
    ) -> Result<(), ModelError> {
        Python::attach(|py| -> PyResult<()> {
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
struct PyModelRunner {
    runner: Arc<Mutex<ModelRunner>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelRunner {
    #[new]
    fn __new__(
        model_path: String,
        provider: Py<ModelProviderABC>,
        pre_fetch_time_ms: Option<u64>,
    ) -> PyResult<Self> {
        let input_provider = Arc::new(PyModelProvider::__new__(provider));

        let runner = ModelRunner::new(
            model_path,
            input_provider,
            pre_fetch_time_ms.map(Duration::from_millis),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            runner: Arc::new(Mutex::new(runner)),
        })
    }

    #[pyo3(name = "run", signature = (dt, total_runtime = None, total_steps = None))]
    fn run(
        &self,
        dt: Duration,
        total_runtime: Option<Duration>,
        total_steps: Option<u64>,
    ) -> PyResult<()> {
        let mut runner = self
            .runner
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        runner
            .run(dt, total_runtime, total_steps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    Ok(())
}

define_stub_info_gatherer!(stub_info);
