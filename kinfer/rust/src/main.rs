use kinfer::{Input, ModelSchema, Output};

mod model;
mod onnx_serializer;
mod serializer;

use model::*;
use onnx_serializer::OnnxSerializer;
use serializer::Serializer;

struct ModelRunner {
    model: ort::Session,
    attached_metadata: std::collections::HashMap<String, String>,
    input_schema: ModelSchema,
    output_schema: ModelSchema,
    input_serializer: OnnxSerializer,
    output_serializer: OnnxSerializer,
}

impl ModelRunner {
    fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = load_onnx_model(model_path)?;
        let session = ort::Session::new(model_path)?;

        // Extract metadata
        let mut attached_metadata = std::collections::HashMap::new();
        let mut input_schema = None;
        let mut output_schema = None;

        for prop in model.metadata_props() {
            if prop.key == KINFER_METADATA_KEY {
                let metadata = P::ModelSchema::from_str(&prop.value)?;
                input_schema = Some(metadata.input_schema);
                output_schema = Some(metadata.output_schema);
            } else {
                attached_metadata.insert(prop.key.clone(), prop.value.clone());
            }
        }

        let input_schema = input_schema.ok_or("kinfer_metadata not found in model metadata")?;
        let output_schema = output_schema.ok_or("kinfer_metadata not found in model metadata")?;

        // Create serializers
        let input_serializer = OnnxSerializer::new(input_schema);
        let output_serializer = OnnxSerializer::new(output_schema);

        Ok(Self {
            model: session,
            attached_metadata,
            input_schema,
            output_schema,
            input_serializer,
            output_serializer,
        })
    }

    fn run(&mut self, inputs: P::Input) -> Result<P::Output, Box<dyn std::error::Error>> {
        let inputs_np = self.input_serializer.serialize_input(&inputs)?;
        let outputs_np = self.model.run(None, &inputs_np)?;
        let outputs = self.output_serializer.deserialize_output(&outputs_np)?;
        Ok(outputs)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    Ok(())
}
