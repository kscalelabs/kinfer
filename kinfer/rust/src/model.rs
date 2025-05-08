use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use ndarray::{Array, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use ort::{session::Session, Error as OrtError};

pub fn load_onnx_model<P: AsRef<Path>>(model_path: P) -> Result<Session, OrtError> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}

pub struct ModelRunner {
    session: Session,
}

impl ModelRunner {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        expected_input_names: HashSet<String>,
        expected_output_names: HashSet<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session = load_onnx_model(model_path)?;

        let input_names = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<HashSet<_>>();
        if input_names != expected_input_names {
            return Err(
                format!("Input names do not match expected names: {:?}", input_names).into(),
            );
        }

        let output_names = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect::<HashSet<_>>();
        if output_names != expected_output_names {
            return Err(format!(
                "Output names do not match expected names: {:?}",
                output_names
            )
            .into());
        }

        Ok(Self { session })
    }

    pub fn run(
        &self,
        inputs: HashMap<String, Array<f32, IxDyn>>,
    ) -> Result<HashMap<String, Array<f32, IxDyn>>, Box<dyn std::error::Error>> {
        let mut input_values = Vec::new();
        for input in &self.session.inputs {
            let input_data = inputs
                .get(&input.name)
                .ok_or_else(|| format!("Missing input: {}", input.name))?;
            let input_value = Value::from_array(input_data.view())?;
            input_values.push((input.name.as_str(), input_value));
        }

        let outputs = self.session.run(input_values)?;
        let mut output_map = HashMap::new();
        for (i, output) in self.session.outputs.iter().enumerate() {
            let output_tensor = outputs[i].try_extract_tensor::<f32>()?;
            output_map.insert(output.name.clone(), output_tensor.view().to_owned());
        }
        Ok(output_map)
    }
}
