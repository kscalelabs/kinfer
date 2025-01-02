use crate::{
    kinfer_proto::{
        self as P, AudioFrameSchema, AudioFrameValue, CameraFrameSchema, CameraFrameValue, DType,
        ImuAccelerometerValue, ImuGyroscopeValue, ImuMagnetometerValue, ImuSchema, ImuValue,
        JointCommandValue, JointCommandsSchema, JointCommandsValue, JointPositionUnit,
        JointPositionValue, JointPositionsSchema, JointPositionsValue, JointTorqueUnit,
        JointTorqueValue, JointTorquesSchema, JointTorquesValue, JointVelocitiesSchema,
        JointVelocitiesValue, JointVelocityUnit, JointVelocityValue, ProtoValue, StateTensorSchema,
        StateTensorValue, TimestampSchema, TimestampValue, ValueSchema, VectorCommandSchema,
        VectorCommandValue,
    },
    onnx_serializer::OnnxSerializer,
    serializer::{
        AudioFrameSerializer, CameraFrameSerializer, ImuSerializer, JointCommandsSerializer,
        JointPositionsSerializer, JointTorquesSerializer, JointVelocitiesSerializer,
        StateTensorSerializer, TimestampSerializer, VectorCommandSerializer,
    },
};

use ndarray::Array;
use ort::value::Value as OrtValue;
use std::f32::consts::PI;

#[test]
fn test_serialize_joint_positions() {
    let joint_names = vec![
        "joint_1".to_string(),
        "joint_2".to_string(),
        "joint_3".to_string(),
    ];
    let schema = ValueSchema {
        value_name: "test".to_string(),
        value_type: Some(P::proto::value_schema::ValueType::JointPositions(
            JointPositionsSchema {
                unit: JointPositionUnit::Degrees as i32,
                joint_names: joint_names.clone(),
            },
        )),
    };

    let serializer = OnnxSerializer::new(schema.clone());

    // Test with matching units
    let value = JointPositionsValue {
        values: vec![
            JointPositionValue {
                joint_name: "joint_1".to_string(),
                value: 60.0,
                unit: JointPositionUnit::Degrees as i32,
            },
            JointPositionValue {
                joint_name: "joint_2".to_string(),
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
        }
        _ => panic!("Wrong schema type"),
    }
    .unwrap();

    // Verify tensor shape and values
    let tensor = result.try_extract_tensor::<f32>().unwrap();
    let array = tensor.view();
    assert_eq!(array.shape(), &[3]);
    assert_eq!(array[[0]], 60.0); // joint_1
    assert_eq!(array[[1]], 30.0); // joint_2
    assert_eq!(array[[2]], 90.0); // joint_3

    let deserialized = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.deserialize_joint_positions(schema, result)
        }
        _ => panic!("Wrong schema type"),
    }
    .unwrap();

    // Verify full deserialization
    assert_eq!(deserialized.values.len(), value.values.len());
    for (expected, actual) in value.values.iter().zip(deserialized.values.iter()) {
        assert_eq!(expected.joint_name, actual.joint_name);
        assert_eq!(expected.value, actual.value);
        assert_eq!(expected.unit, actual.unit);
    }

    // Test unit conversion
    let value_radians = JointPositionsValue {
        values: vec![JointPositionValue {
            joint_name: "joint_1".to_string(),
            value: PI / 6.0,
            unit: JointPositionUnit::Radians as i32,
        }],
    };

    let schema_radians = ValueSchema {
        value_name: "test".to_string(),
        value_type: Some(P::proto::value_schema::ValueType::JointPositions(
            JointPositionsSchema {
                unit: JointPositionUnit::Radians as i32,
                joint_names: vec!["joint_1".to_string()],
            },
        )),
    };

    let serializer = OnnxSerializer::new(schema_radians.clone());
    let result = match schema_radians.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.serialize_joint_positions(schema, value_radians.clone())
        }
        _ => panic!("Wrong schema type"),
    }
    .unwrap();

    let tensor = result.try_extract_tensor::<f32>().unwrap();
    let array = tensor.view();
    assert!((array[[0]] - PI / 6.0).abs() < 1e-6);
}

#[test]
fn test_serialize_joint_positions_errors() {
    let schema = ValueSchema {
        value_name: "test".to_string(),
        value_type: Some(P::proto::value_schema::ValueType::JointPositions(
            JointPositionsSchema {
                unit: JointPositionUnit::Degrees as i32,
                joint_names: vec!["joint_1".to_string(), "joint_2".to_string()],
            },
        )),
    };

    let serializer = OnnxSerializer::new(schema.clone());

    // Test cases that should fail:

    // Case 1: Wrong number of joints
    let value_wrong_count = JointPositionsValue {
        values: vec![JointPositionValue {
            joint_name: "joint_1".to_string(),
            value: 60.0,
            unit: JointPositionUnit::Degrees as i32,
        }],
    };

    let result = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.serialize_joint_positions(schema, value_wrong_count)
        }
        _ => panic!("Wrong schema type"),
    };
    assert!(
        result.is_err(),
        "Should fail when joint count doesn't match"
    );

    // Case 2: Wrong joint names
    let value_wrong_names = JointPositionsValue {
        values: vec![
            JointPositionValue {
                joint_name: "wrong_joint_1".to_string(),
                value: 60.0,
                unit: JointPositionUnit::Degrees as i32,
            },
            JointPositionValue {
                joint_name: "wrong_joint_2".to_string(),
                value: 30.0,
                unit: JointPositionUnit::Degrees as i32,
            },
        ],
    };

    let result = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.serialize_joint_positions(schema, value_wrong_names)
        }
        _ => panic!("Wrong schema type"),
    };
    assert!(result.is_err(), "Should fail when joint names don't match");

    // Case 3: Wrong unit type
    let value_wrong_unit = JointPositionsValue {
        values: vec![
            JointPositionValue {
                joint_name: "joint_1".to_string(),
                value: 60.0,
                unit: JointPositionUnit::Radians as i32,
            },
            JointPositionValue {
                joint_name: "joint_2".to_string(),
                value: 30.0,
                unit: JointPositionUnit::Radians as i32,
            },
        ],
    };

    let result = match schema.value_type.as_ref().unwrap() {
        P::proto::value_schema::ValueType::JointPositions(schema) => {
            serializer.serialize_joint_positions(schema, value_wrong_unit)
        }
        _ => panic!("Wrong schema type"),
    };
    assert!(result.is_err(), "Should fail when units don't match");
}
