use async_trait::async_trait;

use kinfer::model::{ModelProvider, ModelRunner};
use ndarray::{Array, IxDyn};
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::sync::Arc;

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

    fn get_joint_angles(&self, _joint_names: Vec<String>) -> PyResult<Py<PyArrayDyn<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_joint_angles",
        ))
    }

    fn get_joint_angular_velocities(
        &self,
        _joint_names: Vec<String>,
    ) -> PyResult<Py<PyArrayDyn<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_joint_angular_velocities",
        ))
    }

    fn get_projected_gravity(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_projected_gravity",
        ))
    }

    fn get_accelerometer(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_accelerometer",
        ))
    }

    fn get_gyroscope(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        Err(PyNotImplementedError::new_err(
            "Must override get_gyroscope",
        ))
    }

    fn take_action(&self, _action: Py<PyArrayDyn<f32>>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err("Must override take_action"))
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct PyModelProvider {
    obj: Arc<Py<ModelProviderABC>>,
}

impl PyModelProvider {
    async fn call_python_async(
        &self,
        method: &str,
        args: Vec<PyObject>,
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let obj = self.obj.clone();
        let method = method.to_string();

        // Execute the Python call in a blocking thread-safe way
        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<Array<f32, IxDyn>> {
                let obj = obj.bind(py);
                let method = obj.getattr(&method)?;

                // Create tuple properly, unwrapping the Result
                let args_tuple = PyTuple::new(py, args)?;
                let output = method.call1((args_tuple,))?;

                let array = output.extract::<Vec<f32>>()?;
                Ok(Array::from_vec(array).into_dyn())
            })
        })
        .await??;

        Ok(result)
    }
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
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
            Ok(vec![joint_names.to_vec().into_pyobject(py)?.into()])
        })?;
        self.call_python_async("get_joint_angles", args).await
    }

    async fn get_joint_angular_velocities(
        &self,
        joint_names: &[String],
    ) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
            Ok(vec![joint_names.to_vec().into_pyobject(py)?.into()])
        })?;
        self.call_python_async("get_joint_angular_velocities", args)
            .await
    }

    async fn get_projected_gravity(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        self.call_python_async("get_projected_gravity", vec![])
            .await
    }

    async fn get_accelerometer(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        self.call_python_async("get_accelerometer", vec![]).await
    }

    async fn get_gyroscope(&self) -> Result<Array<f32, IxDyn>, Box<dyn std::error::Error>> {
        self.call_python_async("get_gyroscope", vec![]).await
    }

    async fn take_action(
        &self,
        action: Array<f32, IxDyn>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
            let array = numpy::PyArray::from_array(py, &action);
            Ok(vec![array.into()])
        })?;
        self.call_python_async("take_action", args).await?;
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
}

#[pymodule]
fn rust_bindings(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_class::<ModelProviderABC>()?;
    m.add_class::<PyModelRunner>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
