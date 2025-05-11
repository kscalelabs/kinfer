use async_trait::async_trait;
use kinfer::model::{ModelProvider, ModelRunner};
use ndarray::{Array, Ix1, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::sync::Arc;
use tokio::sync::Mutex;

#[pyfunction]
#[gen_stub_pyfunction]
fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pyclass(subclass)]
#[gen_stub_pyclass]
pub struct ModelProviderABC;

#[gen_stub_pymethods]
#[pymethods]
impl ModelProviderABC {
    #[new]
    fn new() -> Self {
        ModelProviderABC
    }

    fn get_joint_angles<'py>(
        &self,
        joint_names: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let n = joint_names.len();
        Err(PyNotImplementedError::new_err(format!(
            "Must override get_joint_angles with {} joint names",
            n
        )))
    }

    fn get_joint_angular_velocities<'py>(
        &self,
        joint_names: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let n = joint_names.len();
        Err(PyNotImplementedError::new_err(format!(
            "Must override get_joint_angular_velocities with {} joint names",
            n
        )))
    }

    fn get_projected_gravity<'py>(&self) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_projected_gravity",
        ))
    }

    fn get_accelerometer<'py>(&self) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_accelerometer",
        ))
    }

    fn get_gyroscope<'py>(&self) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_gyroscope",
        ))
    }

    fn take_action<'py>(&self, action: Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let n = action.len()?;
        Err(PyNotImplementedError::new_err(format!(
            "Must override take_action with {} action",
            n
        )))
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct PyModelProvider {
    obj: Arc<Py<ModelProviderABC>>,
    lock: Arc<Mutex<()>>, // Add mutex for synchronization
}

#[pymethods]
impl PyModelProvider {
    #[new]
    fn new(obj: Py<ModelProviderABC>) -> Self {
        Self {
            obj: Arc::new(obj),
            lock: Arc::new(Mutex::new(())), // Initialize mutex
        }
    }
}

#[async_trait]
impl ModelProvider for PyModelProvider {
    async fn get_joint_angles(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = (joint_names,);
            let result = obj.call_method(py, "get_joint_angles", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })?;
        Ok(args)
    }

    async fn get_joint_angular_velocities(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = (joint_names,);
            let result = obj.call_method(py, "get_joint_angular_velocities", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })?;
        Ok(args)
    }

    async fn get_projected_gravity(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_projected_gravity", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })?;
        Ok(args)
    }

    async fn get_accelerometer(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_accelerometer", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })?;
        Ok(args)
    }

    async fn get_gyroscope(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_gyroscope", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })?;
        Ok(args)
    }

    async fn get_carry(
        &self,
        carry: Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        Ok(carry)
    }

    async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _guard = self.lock.lock().await;
        Python::with_gil(|py| -> PyResult<()> {
            let obj = self.obj.clone();
            let action_1d = action
                .into_dimensionality::<Ix1>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let args = (PyArray1::from_array(py, &action_1d),);
            obj.call_method(py, "take_action", args, None)?;
            Ok(())
        })?;
        Ok(())
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct PyModelRunner {
    runner: Arc<ModelRunner>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelRunner {
    #[new]
    fn new(model_path: String, provider: Py<ModelProviderABC>) -> PyResult<Self> {
        let input_provider = Arc::new(PyModelProvider {
            obj: Arc::new(provider),
            lock: Arc::new(Mutex::new(())),
        });

        let runner = tokio::runtime::Runtime::new().unwrap().block_on(async {
            ModelRunner::new(model_path, input_provider)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })?;

        Ok(Self {
            runner: Arc::new(runner),
        })
    }

    fn init(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        let runner = self.runner.clone();
        let result = tokio::runtime::Runtime::new().unwrap().block_on(async {
            runner
                .init()
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })?;

        Python::with_gil(|py| {
            let array = numpy::PyArray::from_array(py, &result);
            Ok(array.into())
        })
    }

    fn step(
        &self,
        carry: Py<PyArrayDyn<f32>>,
    ) -> PyResult<(Py<PyArrayDyn<f32>>, Py<PyArrayDyn<f32>>)> {
        let runner = self.runner.clone();
        let carry_array = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let carry_array = carry.bind(py);
            Ok(carry_array.to_owned_array())
        })?;

        let result = tokio::runtime::Runtime::new().unwrap().block_on(async {
            runner
                .step(carry_array)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })?;

        Python::with_gil(|py| {
            let (output, carry) = result;
            let output_array = numpy::PyArray::from_array(py, &output);
            let carry_array = numpy::PyArray::from_array(py, &carry);
            Ok((output_array.into(), carry_array.into()))
        })
    }
}

#[pymodule]
fn rust_bindings(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_class::<ModelProviderABC>()?;
    m.add_class::<PyModelRunner>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
