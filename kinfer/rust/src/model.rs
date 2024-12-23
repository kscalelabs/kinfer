use crate::onnx_serializer::OnnxSerializer;
use crate::serializer::Serializer;
use crate::kinfer_proto::{ModelSchema, ProtoIO, ProtoIOSchema};
use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::{session::Session, Error as OrtError};

pub fn load_onnx_model<P: AsRef<Path>>(model_path: P) -> Result<Session, OrtError> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}

const KINFER_METADATA_KEY: &str = "kinfer_metadata";

pub struct ModelRunner {
    session: Session,
    attached_metadata: std::collections::HashMap<String, String>,
    schema: ModelSchema,
    input_serializer: OnnxSerializer,
    output_serializer: OnnxSerializer,
}

impl ModelRunner {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let session = load_onnx_model(model_path)?;
        let mut attached_metadata = std::collections::HashMap::new();

        // Extract metadata and attempt to parse schema
        let mut schema = None;
        let metadata = session.metadata()?;
        for prop in metadata.custom_keys()? {
            if prop == KINFER_METADATA_KEY {
                // TODO: Not implemented yet - need to parse kinfer_metadata from model metadata
                unimplemented!("Parsing kinfer_metadata from model metadata is not yet implemented");
                // schema = Some(ProtoIOSchema::parse_from_str(&metadata.custom(prop.as_str())?)?);
            } else {
                attached_metadata.insert(prop.to_string(), metadata.custom(prop.as_str())?.map_or_else(String::new, |s| s.to_string()));
            }
        }

        let schema: ModelSchema = schema.ok_or_else(|| "kinfer_metadata not found in model metadata")?;

        // Create serializers for input and output
        let input_serializer = OnnxSerializer::new(schema.input_schema.clone());
        let output_serializer = OnnxSerializer::new(schema.output_schema.clone());

        Ok(Self {
            session,
            attached_metadata,
            schema,
            input_serializer,
            output_serializer,
        })
    }

    pub fn run(&self, inputs: ProtoIO) -> Result<ProtoIO, Box<dyn std::error::Error>> {
        let inputs = self.input_serializer.serialize_io(&self.schema.input_schema, inputs)?;

        let inputs_np = inputs.try_extract_tensor()?;
        
        // Run inference
        let outputs_np = self.session.run(&[&inputs_np])?;
        
        // Deserialize outputs from ONNX format
        let outputs = self.output_serializer.deserialize_io(&self.schema.output_schema, outputs_np)?;
        
        Ok(outputs)
    }

    pub fn input_schema(&self) -> &ModelSchema {
        &self.schema.input_schema
    }

    pub fn output_schema(&self) -> &ModelSchema {
        &self.schema.output_schema
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    Ok(())
}
