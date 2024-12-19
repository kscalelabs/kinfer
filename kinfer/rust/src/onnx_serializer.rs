use crate::serializer::{
    convert_position, convert_torque, convert_velocity, AudioFrameSerializer,
    CameraFrameSerializer, ImuSerializer, JointCommandsSerializer, JointPositionsSerializer,
    JointTorquesSerializer, JointVelocitiesSerializer, Serializer, StateTensorSerializer,
    TimestampSerializer, VectorCommandSerializer,
};

use ndarray::{Array, Array1, Array2, Array3, ArrayView2};
use ort::value::{Tensor, Value as OrtValue};
use std::error::Error;

use crate::proto::proto::{
    value::Value, value_schema::ValueType, AudioFrameSchema, AudioFrameValue, CameraFrameSchema,
    CameraFrameValue, DType, ImuAccelerometerValue, ImuGyroscopeValue, ImuMagnetometerValue,
    ImuSchema, ImuValue, JointCommandSchema, JointCommandValue, JointCommandsSchema,
    JointCommandsValue, JointPositionUnit, JointPositionValue, JointPositionsSchema,
    JointPositionsValue, JointTorqueUnit, JointTorqueValue, JointTorquesSchema, JointTorquesValue,
    JointVelocitiesSchema, JointVelocitiesValue, JointVelocityUnit, JointVelocityValue,
    StateTensorSchema, StateTensorValue, TimestampSchema, TimestampValue, Value, ValueSchema,
    VectorCommandSchema, VectorCommandValue,
};

pub struct OnnxSerializer {
    schema: ValueSchema,
}

impl OnnxSerializer {
    pub fn new(schema: ValueSchema) -> Self {
        Self { schema }
    }

    fn array_to_value<T, D>(&self, array: Array<T, D>) -> Result<OrtValue, Box<dyn Error>>
    where
        T: Into<f32> + Copy,
        D: ndarray::Dimension,
    {
        OrtValue::from_array(array.map(|&x| x.into()).into_dyn())
            .map_err(|e| Box::new(e) as Box<dyn Error>)
    }
}

impl JointPositionsSerializer for OnnxSerializer {
    fn serialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: JointPositionsValue,
    ) -> Result<OrtValue, Box<dyn Error>> {
        let mut array = Array1::zeros(schema.joint_names.len());
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(joint) = value.values.iter().find(|v| v.joint_name == *name) {
                array[i] = convert_position(joint.value, joint.unit, schema.unit)?;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: OrtValue,
    ) -> Result<JointPositionsValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor::<f32>()?;
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
    ) -> Result<OrtValue, Box<dyn Error>> {
        let mut array = Array1::zeros(schema.joint_names.len());
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(joint) = value.values.iter().find(|v| v.joint_name == *name) {
                array[i] = convert_velocity(joint.value, joint.unit.clone(), schema.unit.clone())?;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: OrtValue,
    ) -> Result<JointVelocitiesValue, Box<dyn Error>> {
        let tensor: Tensor = value.try_extract()?;
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
                    unit: schema.unit.clone(),
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
    ) -> Result<OrtValue, Box<dyn Error>> {
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
        value: OrtValue,
    ) -> Result<JointTorquesValue, Box<dyn Error>> {
        let tensor: Tensor = value.try_extract()?;
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
                array[[i, 1]] =
                    convert_velocity(cmd.velocity, cmd.velocity_unit, schema.velocity_unit)?;
                array[[i, 2]] =
                    convert_position(cmd.position, cmd.position_unit, schema.position_unit)?;
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
        let tensor: Tensor = value.try_extract()?;
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
            (
                schema.channels as usize,
                schema.height as usize,
                schema.width as usize,
            ),
            bytes.iter().map(|&x| x as f32 / 255.0).collect(),
        )?;
        self.array_to_value(array)
    }

    fn deserialize_camera_frame(
        &self,
        schema: &CameraFrameSchema,
        value: Value,
    ) -> Result<CameraFrameValue, Box<dyn Error>> {
        let tensor: Tensor = value.try_extract()?;
        let array = tensor.view();

        if array.shape()
            != [
                schema.channels as usize,
                schema.height as usize,
                schema.width as usize,
            ]
        {
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
            (schema.channels as usize, schema.sample_rate as usize),
            parse_audio_bytes(&value.data, schema.dtype.clone())?,
        )?;
        self.array_to_value(array)
    }

    fn deserialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: Value,
    ) -> Result<AudioFrameValue, Box<dyn Error>> {
        let tensor: Tensor = value.try_extract()?;
        let array = tensor.view();

        if array.shape() != [schema.channels as usize, schema.sample_rate as usize] {
            return Err("Array shape does not match expected dimensions".into());
        }

        Ok(AudioFrameValue {
            data: audio_array_to_bytes(array, schema.dtype.clone())?,
        })
    }
}

impl ImuSerializer for OnnxSerializer {
    fn serialize_imu(&self, schema: &ImuSchema, value: ImuValue) -> Result<Value, Box<dyn Error>> {
        let mut vectors = Vec::new();

        if schema.use_accelerometer {
            if let Some(acc) = &value.linear_acceleration {
                vectors.push([acc.x, acc.y, acc.z]);
            }
        }
        if schema.use_gyroscope {
            if let Some(gyro) = &value.angular_velocity {
                vectors.push([gyro.x, gyro.y, gyro.z]);
            }
        }
        if schema.use_magnetometer {
            if let Some(mag) = &value.magnetic_field {
                vectors.push([mag.x, mag.y, mag.z]);
            }
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
        let tensor: Tensor = value.try_extract()?;
        let array = tensor.view();
        let mut result = ImuValue::default();
        let mut idx = 0;

        if schema.use_accelerometer {
            result.linear_acceleration = Some(ImuAccelerometerValue {
                x: array[[idx, 0]],
                y: array[[idx, 1]],
                z: array[[idx, 2]],
            });
            idx += 1;
        }
        if schema.use_gyroscope {
            result.angular_velocity = Some(ImuGyroscopeValue {
                x: array[[idx, 0]],
                y: array[[idx, 1]],
                z: array[[idx, 2]],
            });
            idx += 1;
        }
        if schema.use_magnetometer {
            result.magnetic_field = Some(ImuMagnetometerValue {
                x: array[[idx, 0]],
                y: array[[idx, 1]],
                z: array[[idx, 2]],
            });
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
        let tensor: Tensor = value.try_extract()?;
        let total_seconds = tensor.view()[0];
        let elapsed_seconds = total_seconds.trunc() as i64;
        let elapsed_nanos = ((total_seconds.fract() * 1_000_000_000.0).round()) as i32;

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
        let tensor: Tensor = value.try_extract()?;
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
        let shape: Vec<usize> = schema.shape.iter().map(|&x| x as usize).collect();
        let array = Array::from_shape_vec(
            shape,
            parse_tensor_bytes(&value.data, schema.dtype.clone())?,
        )?;
        self.array_to_value(array)
    }

    fn deserialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: Value,
    ) -> Result<StateTensorValue, Box<dyn Error>> {
        let tensor: Tensor = value.try_extract()?;
        let array = tensor.view();

        let expected_shape: Vec<usize> = schema.shape.iter().map(|&x| x as usize).collect();
        if array.shape() != expected_shape.as_slice() {
            return Err("Array shape does not match expected dimensions".into());
        }

        Ok(StateTensorValue {
            data: tensor_array_to_bytes(array, schema.dtype.clone())?,
        })
    }
}

// Helper functions for parsing bytes
fn parse_audio_bytes(_bytes: &[u8], _dtype: DType) -> Result<Vec<f32>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn audio_array_to_bytes(_array: ArrayView2<f32>, _dtype: DType) -> Result<Vec<u8>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn parse_tensor_bytes(_bytes: &[u8], _dtype: DType) -> Result<Vec<f32>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

fn tensor_array_to_bytes(
    _array: ArrayView2<f32>,
    _dtype: DType,
) -> Result<Vec<u8>, Box<dyn Error>> {
    // Implementation needed
    unimplemented!()
}

impl Serializer for OnnxSerializer {
    fn serialize(&self, schema: &ValueSchema, value: Value) -> Result<OrtValue, Box<dyn Error>> {
        match schema.value_type.as_ref().ok_or("Missing value type")? {
            ValueType::JointPositions(ref joint_positions_schema) => match value {
                Value::JointPositions(v) => {
                    self.serialize_joint_positions(joint_positions_schema, v)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::JointVelocities(ref joint_velocities_schema) => match value {
                Value::JointVelocities(v) => {
                    self.serialize_joint_velocities(joint_velocities_schema, v)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::JointTorques(ref joint_torques_schema) => match value {
                Value::JointTorques(v) => self.serialize_joint_torques(joint_torques_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::JointCommands(ref joint_commands_schema) => match value {
                Value::JointCommands(v) => self.serialize_joint_commands(joint_commands_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::CameraFrame(ref camera_frame_schema) => match value {
                Value::CameraFrame(v) => self.serialize_camera_frame(camera_frame_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::AudioFrame(ref audio_frame_schema) => match value {
                Value::AudioFrame(v) => self.serialize_audio_frame(audio_frame_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::Imu(ref imu_schema) => match value {
                Value::Imu(v) => self.serialize_imu(imu_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::Timestamp(ref timestamp_schema) => match value {
                Value::Timestamp(v) => self.serialize_timestamp(timestamp_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::VectorCommand(ref vector_command_schema) => match value {
                Value::VectorCommand(v) => self.serialize_vector_command(vector_command_schema, v),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::StateTensor(ref state_tensor_schema) => match value {
                Value::StateTensor(v) => self.serialize_state_tensor(state_tensor_schema, v),
                _ => Err("Unsupported value type".into()),
            },
        }
    }

    fn deserialize(&self, schema: &ValueSchema, value: OrtValue) -> Result<Value, Box<dyn Error>> {
        match schema.value_type.as_ref().ok_or("Missing value type")? {
            ValueType::JointPositions(ref joint_positions_schema) => Ok(Value::JointPositions(
                self.deserialize_joint_positions(joint_positions_schema, value)?,
            )),
            ValueType::JointVelocities(ref joint_velocities_schema) => Ok(Value::JointVelocities(
                self.deserialize_joint_velocities(joint_velocities_schema, value)?,
            )),
            ValueType::JointTorques(ref joint_torques_schema) => Ok(Value::JointTorques(
                self.deserialize_joint_torques(joint_torques_schema, value)?,
            )),
            ValueType::JointCommands(ref joint_commands_schema) => Ok(Value::JointCommands(
                self.deserialize_joint_commands(joint_commands_schema, value)?,
            )),
            ValueType::CameraFrame(ref camera_frame_schema) => Ok(Value::CameraFrame(
                self.deserialize_camera_frame(camera_frame_schema, value)?,
            )),
            ValueType::AudioFrame(ref audio_frame_schema) => Ok(Value::AudioFrame(
                self.deserialize_audio_frame(audio_frame_schema, value)?,
            )),
            ValueType::Imu(ref imu_schema) => {
                Ok(Value::Imu(self.deserialize_imu(imu_schema, value)?))
            }
            ValueType::Timestamp(ref timestamp_schema) => Ok(Value::Timestamp(
                self.deserialize_timestamp(timestamp_schema, value)?,
            )),
            ValueType::VectorCommand(ref vector_command_schema) => Ok(Value::VectorCommand(
                self.deserialize_vector_command(vector_command_schema, value)?,
            )),
            ValueType::StateTensor(ref state_tensor_schema) => Ok(Value::StateTensor(
                self.deserialize_state_tensor(state_tensor_schema, value)?,
            )),
        }
    }
}
