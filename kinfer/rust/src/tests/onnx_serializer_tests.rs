use crate::{
    kinfer_proto::{
        self as P, AudioFrameSchema, AudioFrameValue, CameraFrameSchema, CameraFrameValue,
        DType, ImuAccelerometerValue, ImuGyroscopeValue, ImuMagnetometerValue, ImuSchema,
        ImuValue, JointCommandsSchema, JointCommandsValue, JointCommandValue,
        JointPositionUnit, JointPositionsSchema, JointPositionsValue, JointPositionValue,
        JointTorqueUnit, JointTorquesSchema, JointTorquesValue, JointTorqueValue,
        JointVelocitiesSchema, JointVelocitiesValue, JointVelocityUnit, JointVelocityValue,
        ProtoValue, StateTensorSchema, StateTensorValue, TimestampSchema, TimestampValue,
        ValueSchema, VectorCommandSchema, VectorCommandValue,
    },
    onnx_serializer::OnnxSerializer,
    serializer::{
        JointPositionsSerializer, JointVelocitiesSerializer, JointTorquesSerializer,
        JointCommandsSerializer, CameraFrameSerializer, AudioFrameSerializer, ImuSerializer,
        TimestampSerializer, VectorCommandSerializer, StateTensorSerializer,
    },
};

use ndarray::Array;
use ort::value::Value as OrtValue;

#[test]
fn test_serialize_joint_positions() {
    let schema = ValueSchema {
        value_name: "test".to_string(),
        value_type: Some(P::proto::value_schema::ValueType::JointPositions(
            JointPositionsSchema {
                unit: JointPositionUnit::Degrees as i32,
                joint_names: vec!["joint_1".to_string(), "joint_2".to_string(), "joint_3".to_string()],
            },
        )),
    };

    let serializer = OnnxSerializer::new(schema.clone());
    
    let value = JointPositionsValue {
        values: vec![
            JointPositionValue {
                joint_name: "joint_2".to_string(),
                value: 60.0,
                unit: JointPositionUnit::Degrees as i32,
            },
            JointPositionValue {
                joint_name: "joint_1".to_string(),
                value: 30.0,
                unit: JointPositionUnit::Degrees as i32,
            },
            JointPositionValue {
                joint_name: "joint_3".to_string(),
                value: 90.0,
                unit: JointPositionUnit::Degrees as i32,
            },
        ],
    };

    let result = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.serialize_joint_positions(schema, value.clone())
        },
        _ => panic!("Wrong schema type"),
    }.unwrap();

    let deserialized = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.deserialize_joint_positions(schema, result)
        },
        _ => panic!("Wrong schema type"),
    }.unwrap();

    assert_eq!(deserialized.values.len(), value.values.len());
}

#[test]
fn test_serialize_camera_frame() {
    let schema = ValueSchema {
        value_name: "test".to_string(),
        value_type: Some(P::proto::value_schema::ValueType::CameraFrame(
            CameraFrameSchema {
                width: 32,
                height: 64,
                channels: 3,
            },
        )),
    };

    let serializer = OnnxSerializer::new(schema.clone());
    
    let data: Vec<u8> = (0..32*64*3).map(|_| rand::random::<u8>()).collect();
    let value = CameraFrameValue {
        data,
    };

    let result = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::CameraFrame(schema) => {
            serializer.serialize_camera_frame(schema, value.clone())
        },
        _ => panic!("Wrong schema type"),
    }.unwrap();

    let deserialized = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::CameraFrame(schema) => {
            serializer.deserialize_camera_frame(schema, result)
        },
        _ => panic!("Wrong schema type"),
    }.unwrap();

    assert_eq!(deserialized.data, value.data);
}

#[test]
fn test_serialize_timestamp() {
    let schema = ValueSchema {
        value_name: "test".to_string(),
        value_type: Some(P::proto::value_schema::ValueType::Timestamp(
            TimestampSchema {
                start_seconds: 0,
                start_nanos: 0,
            },
        )),
    };

    let serializer = OnnxSerializer::new(schema.clone());
    
    let value = TimestampValue {
        seconds: 1,
        nanos: 500_000_000,
    };

    let result = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::Timestamp(schema) => {
            serializer.serialize_timestamp(schema, value.clone())
        },
        _ => panic!("Wrong schema type"),
    }.unwrap();

    let deserialized = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::Timestamp(schema) => {
            serializer.deserialize_timestamp(schema, result)
        },
        _ => panic!("Wrong schema type"),
    }.unwrap();

    assert_eq!(deserialized, value);
}
