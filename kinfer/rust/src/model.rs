use crate::kinfer_proto::{ModelSchema, ProtoIO, ProtoIOSchema};
use crate::onnx_serializer::OnnxMultiSerializer;
use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use prost::Message;
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
    input_serializer: OnnxMultiSerializer,
    output_serializer: OnnxMultiSerializer,
}

impl ModelRunner {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let session = load_onnx_model(model_path)?;
        let mut attached_metadata = std::collections::HashMap::new();

        // Extract metadata and attempt to parse schema
        let mut schema = None;
        {
            let metadata = session.metadata()?;
            for prop in metadata.custom_keys()? {
                if prop == KINFER_METADATA_KEY {
                    let schema_bytes = metadata.custom(prop.as_str())?;
                    if let Some(bytes) = schema_bytes {
                        schema = Some(ModelSchema::decode(&mut bytes.as_bytes())?);
                    }
                } else {
                    attached_metadata.insert(
                        prop.to_string(),
                        metadata
                            .custom(prop.as_str())?
                            .map_or_else(String::new, |s| s.to_string()),
                    );
                }
            }
        }

        let schema: ModelSchema = schema.ok_or_else(|| "kinfer_metadata not found in model metadata")?;

        // Use as_ref() to borrow the Option contents and clone after ok_or
        let input_schema = schema
            .input_schema
            .as_ref()
            .ok_or("Missing input schema")?
            .clone();
        let output_schema = schema
            .output_schema
            .as_ref()
            .ok_or("Missing output schema")?
            .clone();

        // Create serializers for input and output
        let input_serializer = OnnxMultiSerializer::new(input_schema);
        let output_serializer = OnnxMultiSerializer::new(output_schema);

        Ok(Self {
            session,
            attached_metadata,
            schema,
            input_serializer,
            output_serializer,
        })
    }

    pub fn run(&self, inputs: ProtoIO) -> Result<ProtoIO, Box<dyn std::error::Error>> {
        // Serialize inputs to ONNX format
        let inputs = self.input_serializer.serialize_io(inputs)?;

        // Get input names from the session
        let input_names = self
            .session
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>();

        // Create input name-value pairs
        let input_values = vec![(input_names[0], inputs)];

        let outputs = self.session.run(input_values)?;

        let output_values = outputs
            .values()
            .map(|v: ort::value::ValueRef<'_>| {
                v.try_upgrade().map_err(|e| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to upgrade value"),
                    )) as Box<dyn std::error::Error>
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Deserialize outputs from ONNX format
        let outputs = self.output_serializer.deserialize_io(output_values)?;

        Ok(outputs)
    }

    pub fn export_model<P: AsRef<Path>>(&self, model_path: P) -> Result<(), Box<dyn std::error::Error>> {
        let model_bytes = self.session.model_as_bytes()?;
        let mut model = ModelProto::decode(&mut model_bytes.as_slice())?;
        model.set_metadata_props(self.schema.encode_to_vec())?;
        std::fs::write(model_path, model.write_to_bytes()?)?;
        Ok(())
    }

    pub fn input_schema(&self) -> Result<ProtoIOSchema, Box<dyn std::error::Error>> {
        self.schema
            .input_schema
            .as_ref()
            .ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Missing input schema",
                )) as Box<dyn std::error::Error>
            })
            .map(|schema| schema.clone())
    }

    pub fn output_schema(&self) -> Result<ProtoIOSchema, Box<dyn std::error::Error>> {
        self.schema
            .output_schema
            .as_ref()
            .ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Missing output schema",
                )) as Box<dyn std::error::Error>
            })
            .map(|schema| schema.clone())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    Ok(())
}
