use ort::{
    Environment,
    ExecutionProvider,
    GraphOptimizationLevel,
    Session,
    SessionBuilder
};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;
use ort::Value;

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub type_info: String,
}

#[derive(Debug)]
pub struct ONNXModel {
    session: Session,
    metadata: HashMap<String, JsonValue>,
    input_details: Vec<TensorInfo>,
    output_details: Vec<TensorInfo>,
}

#[derive(Debug)]
pub enum InputData<'a> {
    Single(Value<'a>),
    List(Vec<Value<'a>>),
    Map(Vec<(&'a str, Value<'a>)>),
}

#[derive(Debug)]
pub enum OutputData<'a> {
    Single(Value<'a>),
    List(Vec<Value<'a>>),
    Map(Vec<(String, Value<'a>)>),
}

impl ONNXModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize environment
        let environment = Environment::builder()
            .with_name("onnx_environment")
            .with_execution_providers(vec![ExecutionProvider::CPU(Default::default())])
            .build()?
            .into_arc();

        // Create session
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;
        
        // Extract metadata from the model
        let mut metadata = HashMap::new();
        
        // Get input and output details
        let input_details = session
            .inputs
            .iter()
            .map(|input| TensorInfo {
                name: input.name.clone(),
                shape: input.dimensions().map(|d| d.unwrap_or(-1) as i64).collect(),
                type_info: format!("{:?}", input.input_type),
            })
            .collect();

        let output_details = session
            .outputs
            .iter()
            .map(|output| TensorInfo {
                name: output.name.clone(),
                shape: output.dimensions().map(|d| d.unwrap_or(-1) as i64).collect(),
                type_info: format!("{:?}", output.output_type),
            })
            .collect();

        Ok(Self {
            session,
            metadata,
            input_details,
            output_details,
        })
    }

    pub fn get_metadata(&self) -> &HashMap<String, JsonValue> {
        &self.metadata
    }

    pub fn get_input_details(&self) -> &[TensorInfo] {
        &self.input_details
    }

    pub fn get_output_details(&self) -> &[TensorInfo] {
        &self.output_details
    }
}

impl<'a> From<Value<'a>> for InputData<'a> {
    fn from(tensor: Value<'a>) -> Self {
        InputData::Single(tensor)
    }
}

impl<'a> From<Vec<Value<'a>>> for InputData<'a> {
    fn from(tensors: Vec<Value<'a>>) -> Self {
        InputData::List(tensors)
    }
}

impl<'a> From<Vec<(&'a str, Value<'a>)>> for InputData<'a> {
    fn from(named_tensors: Vec<(&'a str, Value<'a>)>) -> Self {
        InputData::Map(named_tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2};
    use ort::{Environment, ExecutionProvider};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn create_test_model() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        // Create a temporary file that automatically gets cleaned up
        let model_file = NamedTempFile::new()?;
        
        // Use Python to create and export a test model
        let python_code = format!(
            r#"
import torch
import torch.nn as nn
import onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

# Create model and add metadata
model = SimpleModel()
dummy_input = torch.randn(1, 10)

# Add metadata
metadata = {{"test_key": "test_value"}}
import json
meta = onnx.StringStringEntryProto()
meta.key = "kinfer_metadata"
meta.value = json.dumps(metadata)

# Export model
torch.onnx.export(model, dummy_input, '{}')
model_proto = onnx.load('{}')
model_proto.metadata_props.append(meta)
onnx.save(model_proto, '{}')
"#,
            model_file.path().display(),
            model_file.path().display(),
            model_file.path().display()
        );
        
        let status = std::process::Command::new("python")
            .arg("-c")
            .arg(python_code)
            .status()?;
            
        assert!(status.success());
        Ok(model_file)
    }

    #[test]
    fn test_model_inference() -> Result<(), Box<dyn std::error::Error>> {
        // Create test model file
        let model_file = create_test_model()?;
        
        // Initialize model
        let model = ONNXModel::new(model_file.path())?;
        
        // Create dummy input data (batch_size=1, features=10)
        let input_data: Array2<f32> = Array::zeros((1, 10));
        
        // Create input tensor with environment-aware allocator
        let input_tensor = Value::from_array(model.session.allocator(), &input_data)?;
        
        // Run inference
        let outputs = model.run(input_tensor)?;
        
        // Verify output
        match outputs {
            OutputData::Single(output) => {
                if let Value::Tensor(tensor) = output {
                    let shape = tensor.shape();
                    assert_eq!(shape, &[1, 2]);
                    
                    // Optional: verify the output data
                    if let Some(data) = tensor.try_extract::<f32>() {
                        assert_eq!(data.len(), 2); // Total elements should be 2 (1x2)
                    }
                } else {
                    panic!("Expected tensor output");
                }
            },
            _ => panic!("Expected single output tensor"),
        }
        
        Ok(())
    }
}
