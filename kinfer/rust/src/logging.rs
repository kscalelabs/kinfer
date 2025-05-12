use ndarray::{Array, IxDyn};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoggerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub struct NumpyLogger {
    file: File,
}

impl NumpyLogger {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, LoggerError> {
        let file = File::create(path)?;
        Ok(Self { file })
    }

    pub fn log_array(&mut self, array: &Array<f32, IxDyn>) -> Result<(), LoggerError> {
        // Write array shape
        let shape = array.shape();
        self.file.write_all(format!("{:?}\n", shape).as_bytes())?;

        // Write array data
        for value in array.iter() {
            self.file.write_all(format!("{}\n", value).as_bytes())?;
        }
        self.file.flush()?;
        Ok(())
    }
}

impl Drop for NumpyLogger {
    fn drop(&mut self) {
        let _ = self.file.flush();
    }
}
