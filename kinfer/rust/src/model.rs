use ort::{Error as OrtError, GraphOptimizationLevel, Session};

pub fn load_onnx_model(model_path: &str) -> Result<Session, OrtError> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}
