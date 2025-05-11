use async_trait::async_trait;
use kinfer::model::{ModelError, ModelProvider, ModelRunner};
use kinfer::runtime::ModelRuntime;
use ndarray::{Array, Ix1, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::sync::Arc;
use std::sync::Mutex;

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

    fn get_command<'py>(&self) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Err(PyNotImplementedError::new_err("Must override get_command"))
    }

    fn take_action<'py>(
        &self,
        joint_names: Vec<String>,
        action: Bound<'py, PyArray1<f32>>,
    ) -> PyResult<()> {
        let n = action.len()?;
        assert_eq!(joint_names.len(), n);
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
}

#[pymethods]
impl PyModelProvider {
    #[new]
    fn new(obj: Py<ModelProviderABC>) -> Self {
        Self { obj: Arc::new(obj) }
    }
}

#[async_trait]
impl ModelProvider for PyModelProvider {
    async fn get_joint_angles(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, ModelError> {
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = (joint_names,);
            let result = obj.call_method(py, "get_joint_angles", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(args)
    }

    async fn get_joint_angular_velocities(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, ModelError> {
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = (joint_names,);
            let result = obj.call_method(py, "get_joint_angular_velocities", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(args)
    }

    async fn get_projected_gravity(&self) -> Result<Array<f32, IxDyn>, ModelError> {
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_projected_gravity", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(args)
    }

    async fn get_accelerometer(&self) -> Result<Array<f32, IxDyn>, ModelError> {
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_accelerometer", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(args)
    }

    async fn get_gyroscope(&self) -> Result<Array<f32, IxDyn>, ModelError> {
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_gyroscope", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(args)
    }

    async fn get_command(&self) -> Result<Array<f32, IxDyn>, ModelError> {
        let args = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let obj = self.obj.clone();
            let args = ();
            let result = obj.call_method(py, "get_command", args, None)?;
            let array = result.extract::<Vec<f32>>(py)?;
            Ok(Array::from_vec(array).into_dyn())
        })
        .map_err(|e| ModelError::Provider(e.to_string()))?;
        Ok(args)
    }

    async fn get_carry(&self, carry: Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>, ModelError> {
        Ok(carry)
    }

    async fn take_action(
        &self,
        joint_names: Vec<String>,
        action: Array<f32, IxDyn>,
    ) -> Result<(), ModelError> {
        Python::with_gil(|py| -> PyResult<()> {
            let obj = self.obj.clone();
            let action_1d = action
                .into_dimensionality::<Ix1>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let args = (joint_names, PyArray1::from_array(py, &action_1d));
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
}

#[gen_stub_pymethods]
#[pymethods]
impl PyModelRunner {
    #[new]
    fn new(model_path: String, provider: Py<ModelProviderABC>) -> PyResult<Self> {
        let input_provider = Arc::new(PyModelProvider {
            obj: Arc::new(provider),
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

    fn take_action(&self, action: Py<PyArrayDyn<f32>>) -> PyResult<()> {
        let runner = self.runner.clone();
        let action_array = Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
            let action_array = action.bind(py);
            Ok(action_array.to_owned_array())
        })?;

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            runner
                .take_action(action_array)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })?;

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
    fn new(model_runner: PyModelRunner, dt: u64) -> PyResult<Self> {
        Ok(Self {
            runtime: Arc::new(Mutex::new(ModelRuntime::new(model_runner.runner, dt))),
        })
    }

    fn set_slowdown_factor(&self, slowdown_factor: i32) {
        let mut runtime = self.runtime.lock().unwrap();
        runtime.set_slowdown_factor(slowdown_factor);
    }

    fn set_magnitude_factor(&self, magnitude_factor: f32) {
        let mut runtime = self.runtime.lock().unwrap();
        runtime.set_magnitude_factor(magnitude_factor);
    }

    fn start(&self) -> PyResult<()> {
        let mut runtime = self.runtime.lock().unwrap();
        runtime
            .start()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn stop(&self) {
        let mut runtime = self.runtime.lock().unwrap();
        runtime.stop();
    }
}

#[pymodule]
fn rust_bindings(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_class::<ModelProviderABC>()?;
    m.add_class::<PyModelRunner>()?;
    m.add_class::<PyModelRuntime>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
