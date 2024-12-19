use kinfer::{Input, ModelSchema, Output};

use ndarray::Array1;
use ort::tensor::OrtOwnedTensor;
use ort::Value;
use std::collections::HashMap;

pub struct Serializer {
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl Serializer {
    fn new(input_names: Vec<String>, output_names: Vec<String>) -> Self {
        Self {
            input_names,
            output_names,
        }
    }

    fn serialize_inputs(
        &self,
        input: Input
    ) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let mut ort_inputs = Vec::new();
        for name in &self.input_names {
            if let Some(input) = inputs.get(name.as_str()) {
                ort_inputs.push(Value::from_array(input.clone())?);
            } else {
                return Err(format!("Input {} not found", name).into());
            }
        }
        Ok(ort_inputs)
    }

    fn deserialize_outputs(
        &self,
        output: Output
    ) -> HashMap<String, Array1<f32>> {
        let mut deserialized_outputs = HashMap::new();
        for (i, output) in outputs.into_iter().enumerate() {
            if let Some(name) = self.output_names.get(i) {
                deserialized_outputs.insert(name.clone(), output.into_array()?);
            }
        }
        deserialized_outputs
    }
}
