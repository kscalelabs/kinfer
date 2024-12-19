use crate::serializer::{
    AudioFrameSerializer, CameraFrameSerializer, ImuSerializer, JointCommandsSerializer,
    JointPositionsSerializer, JointTorquesSerializer, JointVelocitiesSerializer, Serializer,
    StateTensorSerializer, TimestampSerializer, VectorCommandSerializer,
};

use ndarray::{Array, Array1, Array2, Array3, ArrayView2};
use ort::{
    session::Session,
    tensor::{OrtOwnedTensor, TensorElementType},
    value::{Value, ValueType},
};
use std::error::Error;

use crate::proto::*;

pub struct OnnxSerializer {
    schema: ValueSchema,
}

impl OnnxSerializer {
    pub fn new(schema: ValueSchema) -> Self {
        Self { schema }
    }

    fn array_to_value<T, D>(&self, array: Array<T, D>) -> Result<Value, Box<dyn Error>>
    where
        T: TensorElementType,
        D: ndarray::Dimension,
    {
        Ok(Value::from_array(array.into_dyn())?)
    }
}

impl JointPositionsSerializer for OnnxSerializer {
    fn serialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: JointPositionsValue,
    ) -> Result<Value, Box<dyn Error>> {
        let mut array = Array1::zeros(schema.joint_names.len());
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(joint) = value.values.iter().find(|v| v.joint_name == *name) {
                array[i] = convert_angular_position(joint.value, joint.unit, schema.unit);
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: Value,
    ) -> Result<JointPositionsValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.len() != schema.joint_names.len() {
            return Err("Array length does not match number of joints".into());
        }

        Ok(JointPositionsValue {
            values: schema
                .joint_names
                .iter()
                .enumerate()
                .map(|(i, name)| JointPositionValue {
                    joint_name: name.clone(),
                    value: array[i],
                    unit: schema.unit,
                })
                .collect(),
        })
    }
}

impl JointVelocitiesSerializer for OnnxSerializer {
    fn serialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: JointVelocitiesValue,
    ) -> Result<Value, Box<dyn Error>> {
        let mut array = Array1::zeros(schema.joint_names.len());
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(joint) = value.values.iter().find(|v| v.joint_name == *name) {
                array[i] = convert_angular_velocity(joint.value, joint.unit, schema.unit);
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: Value,
    ) -> Result<JointVelocitiesValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.len() != schema.joint_names.len() {
            return Err("Array length does not match number of joints".into());
        }

        Ok(JointVelocitiesValue {
            values: schema
                .joint_names
                .iter()
                .enumerate()
                .map(|(i, name)| JointVelocityValue {
                    joint_name: name.clone(),
                    value: array[i],
                    unit: schema.unit,
                })
                .collect(),
        })
    }
}

impl JointTorquesSerializer for OnnxSerializer {
    fn serialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: JointTorquesValue,
    ) -> Result<Value, Box<dyn Error>> {
        let mut array = Array1::zeros(schema.joint_names.len());
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(joint) = value.values.iter().find(|v| v.joint_name == *name) {
                array[i] = convert_torque(joint.value, joint.unit, schema.unit);
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: Value,
    ) -> Result<JointTorquesValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.len() != schema.joint_names.len() {
            return Err("Array length does not match number of joints".into());
        }

        Ok(JointTorquesValue {
            values: schema
                .joint_names
                .iter()
                .enumerate()
                .map(|(i, name)| JointTorqueValue {
                    joint_name: name.clone(),
                    value: array[i],
                    unit: schema.unit,
                })
                .collect(),
        })
    }
}

impl JointCommandsSerializer for OnnxSerializer {
    fn serialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: JointCommandsValue,
    ) -> Result<Value, Box<dyn Error>> {
        let mut array = Array2::zeros((schema.joint_names.len(), 5));
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(cmd) = value.values.iter().find(|v| v.joint_name == *name) {
                array[[i, 0]] = convert_torque(cmd.torque, cmd.torque_unit, schema.torque_unit)?;
                array[[i, 1]] = convert_angular_velocity(
                    cmd.velocity,
                    cmd.velocity_unit,
                    schema.velocity_unit,
                )?;
                array[[i, 2]] = convert_angular_position(
                    cmd.position,
                    cmd.position_unit,
                    schema.position_unit,
                )?;
                array[[i, 3]] = cmd.kp;
                array[[i, 4]] = cmd.kd;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: Value,
    ) -> Result<JointCommandsValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.shape() != [schema.joint_names.len(), 5] {
            return Err("Array shape does not match expected dimensions".into());
        }

        Ok(JointCommandsValue {
            values: schema
                .joint_names
                .iter()
                .enumerate()
                .map(|(i, name)| JointCommandValue {
                    joint_name: name.clone(),
                    torque: array[[i, 0]],
                    velocity: array[[i, 1]],
                    position: array[[i, 2]],
                    kp: array[[i, 3]],
                    kd: array[[i, 4]],
                    torque_unit: schema.torque_unit,
                    velocity_unit: schema.velocity_unit,
                    position_unit: schema.position_unit,
                })
                .collect(),
        })
    }
}

impl CameraFrameSerializer for OnnxSerializer {
    fn serialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: CameraFrameValue,
    ) -> Result<Value, Box<dyn Error>> {
        let bytes = value.data;
        let array = Array3::from_shape_vec(
            (schema.channels, schema.height, schema.width),
            bytes.iter().map(|&x| x as f32 / 255.0).collect(),
        )?;
        self.array_to_value(array)
    }

    fn deserialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: Value,
    ) -> Result<CameraFrameValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.shape() != [schema.channels, schema.height, schema.width] {
            return Err("Array shape does not match expected dimensions".into());
        }

        let bytes: Vec<u8> = array
            .iter()
            .map(|&x| (x * 255.0).clamp(0.0, 255.0) as u8)
            .collect();

        Ok(CameraFrameValue { data: bytes })
    }
}

impl AudioFrameSerializer for OnnxSerializer {
    fn serialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: AudioFrameValue,
    ) -> Result<Value, Box<dyn Error>> {
        let array = Array2::from_shape_vec(
            (schema.channels, schema.sample_rate),
            parse_audio_bytes(&value.data, schema.dtype)?,
        )?;
        self.array_to_value(array)
    }

    fn deserialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: Value,
    ) -> Result<AudioFrameValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.shape() != [schema.channels, schema.sample_rate] {
            return Err("Array shape does not match expected dimensions".into());
        }

        Ok(AudioFrameValue {
            data: audio_array_to_bytes(array, schema.dtype)?,
        })
    }
}

impl ImuSerializer for OnnxSerializer {
    fn serialize_imu(&self, schema: &ImuSchema, value: ImuValue) -> Result<Value, Box<dyn Error>> {
        let mut vectors = Vec::new();

        if schema.use_accelerometer {
            vectors.push([
                value.linear_acceleration.x,
                value.linear_acceleration.y,
                value.linear_acceleration.z,
            ]);
        }
        if schema.use_gyroscope {
            vectors.push([
                value.angular_velocity.x,
                value.angular_velocity.y,
                value.angular_velocity.z,
            ]);
        }
        if schema.use_magnetometer {
            vectors.push([
                value.magnetic_field.x,
                value.magnetic_field.y,
                value.magnetic_field.z,
            ]);
        }

        let array = Array2::from_shape_vec(
            (vectors.len(), 3),
            vectors.into_iter().flat_map(|v| v.into_iter()).collect(),
        )?;
        self.array_to_value(array)
    }

    fn deserialize_imu(
        &self,
        schema: &ImuSchema,
        value: Value,
    ) -> Result<ImuValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();
        let mut result = ImuValue::default();
        let mut idx = 0;

        if schema.use_accelerometer {
            result.linear_acceleration.x = array[[idx, 0]];
            result.linear_acceleration.y = array[[idx, 1]];
            result.linear_acceleration.z = array[[idx, 2]];
            idx += 1;
        }
        if schema.use_gyroscope {
            result.angular_velocity.x = array[[idx, 0]];
            result.angular_velocity.y = array[[idx, 1]];
            result.angular_velocity.z = array[[idx, 2]];
            idx += 1;
        }
        if schema.use_magnetometer {
            result.magnetic_field.x = array[[idx, 0]];
            result.magnetic_field.y = array[[idx, 1]];
            result.magnetic_field.z = array[[idx, 2]];
        }

        Ok(result)
    }
}

impl TimestampSerializer for OnnxSerializer {
    fn serialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: TimestampValue,
    ) -> Result<Value, Box<dyn Error>> {
        let elapsed_seconds = value.seconds - schema.start_seconds;
        let elapsed_nanos = value.nanos - schema.start_nanos;
        let total_seconds = elapsed_seconds as f32 + (elapsed_nanos as f32 / 1_000_000_000.0);
        self.array_to_value(Array1::from_vec(vec![total_seconds]))
    }

    fn deserialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: Value,
    ) -> Result<TimestampValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let total_seconds = tensor.view()[0];
        let elapsed_seconds = total_seconds.floor() as i64;
        let elapsed_nanos = ((total_seconds - elapsed_seconds as f32) * 1_000_000_000.0) as i32;

        Ok(TimestampValue {
            seconds: schema.start_seconds + elapsed_seconds,
            nanos: schema.start_nanos + elapsed_nanos,
        })
    }
}

impl VectorCommandSerializer for OnnxSerializer {
    fn serialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: VectorCommandValue,
    ) -> Result<Value, Box<dyn Error>> {
        let array = Array1::from_vec(value.values);
        self.array_to_value(array)
    }

    fn deserialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: Value,
    ) -> Result<VectorCommandValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.len() != schema.dimensions {
            return Err("Array length does not match expected dimensions".into());
        }

        Ok(VectorCommandValue {
            values: array.to_vec(),
        })
    }
}

impl StateTensorSerializer for OnnxSerializer {
    fn serialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: StateTensorValue,
    ) -> Result<Value, Box<dyn Error>> {
        let array = Array::from_shape_vec(
            schema.shape.clone(),
            parse_tensor_bytes(&value.data, schema.dtype)?,
        )?;
        self.array_to_value(array)
    }

    fn deserialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: Value,
    ) -> Result<StateTensorValue, Box<dyn Error>> {
        let tensor: OrtOwnedTensor<f32> = value.try_extract()?;
        let array = tensor.view();

        if array.shape() != schema.shape.as_slice() {
            return Err("Array shape does not match expected dimensions".into());
        }

        Ok(StateTensorValue {
            data: tensor_array_to_bytes(array, schema.dtype)?,
        })
    }
}

// Helper functions for audio and tensor data conversion
fn parse_audio_bytes(bytes: &[u8], dtype: DType) -> Result<Vec<f32>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn audio_array_to_bytes(array: ArrayView2<f32>, dtype: DType) -> Result<Vec<u8>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn parse_tensor_bytes(bytes: &[u8], dtype: DType) -> Result<Vec<f32>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn tensor_array_to_bytes(array: ArrayView2<f32>, dtype: DType) -> Result<Vec<u8>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn convert_torque(
    value: f32,
    from_unit: JointTorqueUnit,
    to_unit: JointTorqueUnit,
) -> Result<f32, Box<dyn Error>> {
    match (from_unit, to_unit) {
        (JointTorqueUnit::NEWTON_METERS, JointTorqueUnit::NEWTON_METERS) => Ok(value),
        _ => Err("Unsupported torque unit".into()),
    }
}

impl Serializer for OnnxSerializer {
    fn serialize(&self, schema: &ValueSchema, value: Value) -> Result<Value, Box<dyn Error>> {
        match schema.value_type {
            ValueType::JointPositions => {
                self.serialize_joint_positions(&schema.joint_positions, value.joint_positions)
            }
            ValueType::JointVelocities => {
                self.serialize_joint_velocities(&schema.joint_velocities, value.joint_velocities)
            }
            // Add other cases...
            _ => Err("Unsupported value type".into()),
        }
    }

    fn deserialize(&self, schema: &ValueSchema, value: Value) -> Result<Value, Box<dyn Error>> {
        match schema.value_type {
            ValueType::JointPositions => Ok(Value::JointPositions(
                self.deserialize_joint_positions(&schema.joint_positions, value)?,
            )),
            ValueType::JointVelocities => Ok(Value::JointVelocities(
                self.deserialize_joint_velocities(&schema.joint_velocities, value)?,
            )),
            // Add other cases...
            _ => Err("Unsupported value type".into()),
        }
    }
}

// Helper functions
fn convert_angular_position(
    value: f32,
    from_unit: JointPositionUnit,
    to_unit: JointPositionUnit,
) -> Result<f32, Box<dyn Error>> {
    match (from_unit, to_unit) {
        (JointPositionUnit::RADIANS, JointPositionUnit::DEGREES) => {
            Ok(value * 180.0 / std::f32::consts::PI)
        }
        (JointPositionUnit::DEGREES, JointPositionUnit::RADIANS) => {
            Ok(value * std::f32::consts::PI / 180.0)
        }
        _ => Err("Unsupported position unit".into()),
    }
}

fn convert_angular_velocity(
    value: f32,
    from_unit: JointVelocityUnit,
    to_unit: JointVelocityUnit,
) -> Result<f32, Box<dyn Error>> {
    match (from_unit, to_unit) {
        (JointVelocityUnit::RADIANS_PER_SECOND, JointVelocityUnit::DEGREES_PER_SECOND) => {
            Ok(value * 180.0 / std::f32::consts::PI)
        }
        (JointVelocityUnit::DEGREES_PER_SECOND, JointVelocityUnit::RADIANS_PER_SECOND) => {
            Ok(value * std::f32::consts::PI / 180.0)
        }
        _ => Err("Unsupported velocity unit".into()),
    }
}
