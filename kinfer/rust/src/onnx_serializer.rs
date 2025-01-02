use crate::serializer::{
    convert_position, convert_torque, convert_velocity, AudioFrameSerializer,
    CameraFrameSerializer, ImuSerializer, JointCommandsSerializer, JointPositionsSerializer,
    JointTorquesSerializer, JointVelocitiesSerializer, Serializer, StateTensorSerializer,
    TimestampSerializer, VectorCommandSerializer,
};

use ndarray::{s, Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2};
use ort::value::{Tensor, Value as OrtValue};
use std::error::Error;

// Import the re-exported types
use crate::kinfer_proto::{
    AudioFrameSchema, AudioFrameValue, CameraFrameSchema, CameraFrameValue, DType,
    ImuAccelerometerValue, ImuGyroscopeValue, ImuMagnetometerValue, ImuSchema, ImuValue,
    JointCommandValue, JointCommandsSchema, JointCommandsValue, JointPositionUnit,
    JointPositionValue, JointPositionsSchema, JointPositionsValue, JointTorqueUnit,
    JointTorqueValue, JointTorquesSchema, JointTorquesValue, JointVelocitiesSchema,
    JointVelocitiesValue, JointVelocityUnit, JointVelocityValue, ProtoIO, ProtoIOSchema,
    ProtoValue, StateTensorSchema, StateTensorValue, TimestampSchema, TimestampValue, ValueSchema,
    VectorCommandSchema, VectorCommandValue,
};

// Import the nested types
use crate::kinfer_proto::proto::value::Value as EnumValue;
use crate::kinfer_proto::proto::value_schema::ValueType;

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
        let array = array.map(|&x| x.into());
        Tensor::from_array(array.into_dyn())
            .map(|tensor| tensor.into_dyn())
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
                let from_unit = JointPositionUnit::try_from(joint.unit)?;
                let to_unit = JointPositionUnit::try_from(schema.unit)?;
                array[i] = convert_position(joint.value, from_unit, to_unit)?;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_positions(
        &self,
        schema: &JointPositionsSchema,
        value: OrtValue,
    ) -> Result<JointPositionsValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
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
                let from_unit = JointVelocityUnit::try_from(joint.unit)?;
                let to_unit = JointVelocityUnit::try_from(schema.unit)?;
                array[i] = convert_velocity(joint.value, from_unit, to_unit)?;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_velocities(
        &self,
        schema: &JointVelocitiesSchema,
        value: OrtValue,
    ) -> Result<JointVelocitiesValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
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
                let from_unit = JointTorqueUnit::try_from(joint.unit)?;
                let to_unit = JointTorqueUnit::try_from(schema.unit)?;
                array[i] = convert_torque(joint.value, from_unit, to_unit)?;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_torques(
        &self,
        schema: &JointTorquesSchema,
        value: OrtValue,
    ) -> Result<JointTorquesValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
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
    ) -> Result<OrtValue, Box<dyn Error>> {
        let mut array = Array2::zeros((schema.joint_names.len(), 5));
        for (i, name) in schema.joint_names.iter().enumerate() {
            if let Some(cmd) = value.values.iter().find(|v| v.joint_name == *name) {
                let cmd_torque_unit = JointTorqueUnit::try_from(cmd.torque_unit)?;
                let cmd_velocity_unit = JointVelocityUnit::try_from(cmd.velocity_unit)?;
                let cmd_position_unit = JointPositionUnit::try_from(cmd.position_unit)?;
                let schema_torque_unit = JointTorqueUnit::try_from(schema.torque_unit)?;
                let schema_velocity_unit = JointVelocityUnit::try_from(schema.velocity_unit)?;
                let schema_position_unit = JointPositionUnit::try_from(schema.position_unit)?;
                array[[i, 0]] = convert_torque(cmd.torque, cmd_torque_unit, schema_torque_unit)?;
                array[[i, 1]] =
                    convert_velocity(cmd.velocity, cmd_velocity_unit, schema_velocity_unit)?;
                array[[i, 2]] =
                    convert_position(cmd.position, cmd_position_unit, schema_position_unit)?;
                array[[i, 3]] = cmd.kp;
                array[[i, 4]] = cmd.kd;
            }
        }
        self.array_to_value(array)
    }

    fn deserialize_joint_commands(
        &self,
        schema: &JointCommandsSchema,
        value: OrtValue,
    ) -> Result<JointCommandsValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
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
    ) -> Result<OrtValue, Box<dyn Error>> {
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
        value: OrtValue,
    ) -> Result<CameraFrameValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
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
            .map(|&x: &f32| (x * 255.0).clamp(0.0, 255.0) as u8)
            .collect();

        Ok(CameraFrameValue { data: bytes })
    }
}

impl AudioFrameSerializer for OnnxSerializer {
    fn serialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: AudioFrameValue,
    ) -> Result<OrtValue, Box<dyn Error>> {
        let array = Array2::from_shape_vec(
            (schema.channels as usize, schema.sample_rate as usize),
            parse_audio_bytes(&value.data, schema.dtype.try_into()?)?,
        )?;
        self.array_to_value(array)
    }

    fn deserialize_audio_frame(
        &self,
        schema: &AudioFrameSchema,
        value: OrtValue,
    ) -> Result<AudioFrameValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
        let array = tensor.view();

        if array.shape() != [schema.channels as usize, schema.sample_rate as usize] {
            return Err("Array shape does not match expected dimensions".into());
        }

        let array = array.into_dimensionality::<ndarray::Ix2>()?;

        Ok(AudioFrameValue {
            data: audio_array_to_bytes(array, schema.dtype.try_into()?)?,
        })
    }
}

impl ImuSerializer for OnnxSerializer {
    fn serialize_imu(
        &self,
        schema: &ImuSchema,
        value: ImuValue,
    ) -> Result<OrtValue, Box<dyn Error>> {
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
        value: OrtValue,
    ) -> Result<ImuValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
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
    ) -> Result<OrtValue, Box<dyn Error>> {
        let elapsed_seconds = value.seconds - schema.start_seconds;
        let elapsed_nanos = value.nanos - schema.start_nanos;
        let total_seconds = elapsed_seconds as f32 + (elapsed_nanos as f32 / 1_000_000_000.0);
        self.array_to_value(Array1::from_vec(vec![total_seconds]))
    }

    fn deserialize_timestamp(
        &self,
        schema: &TimestampSchema,
        value: OrtValue,
    ) -> Result<TimestampValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
        let array = tensor.view();

        // Get first element using iterator
        let total_seconds: f32 = *array.iter().next().ok_or("Timestamp tensor is empty")?;

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
    ) -> Result<OrtValue, Box<dyn Error>> {
        let array = Array1::from_vec(value.values);
        self.array_to_value(array)
    }

    fn deserialize_vector_command(
        &self,
        schema: &VectorCommandSchema,
        value: OrtValue,
    ) -> Result<VectorCommandValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
        let array = tensor.view();

        if array.len() != schema.dimensions as usize {
            return Err("Array length does not match expected dimensions".into());
        }

        Ok(VectorCommandValue {
            values: array.iter().copied().collect(),
        })
    }
}

impl StateTensorSerializer for OnnxSerializer {
    fn serialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: StateTensorValue,
    ) -> Result<OrtValue, Box<dyn Error>> {
        let shape: Vec<usize> = schema.shape.iter().map(|&x| x as usize).collect();
        let array = Array::from_shape_vec(
            shape,
            parse_tensor_bytes(&value.data, schema.dtype.try_into()?)?,
        )?;
        self.array_to_value(array)
    }

    fn deserialize_state_tensor(
        &self,
        schema: &StateTensorSchema,
        value: OrtValue,
    ) -> Result<StateTensorValue, Box<dyn Error>> {
        let tensor = value.try_extract_tensor()?;
        let array = tensor.view();

        let expected_shape: Vec<usize> = schema.shape.iter().map(|&x| x as usize).collect();
        if array.shape() != expected_shape.as_slice() {
            return Err("Array shape does not match expected dimensions".into());
        }

        Ok(StateTensorValue {
            data: tensor_array_to_bytes(array.view(), schema.dtype.try_into()?)?,
        })
    }
}

// Helper functions for parsing bytes
fn parse_audio_bytes(bytes: &[u8], dtype: DType) -> Result<Vec<f32>, Box<dyn Error>> {
    match dtype {
        DType::Fp32 => {
            let mut result = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result.push(value);
            }
            Ok(result)
        }
        _ => Err("Unsupported audio data type".into()),
    }
}

fn audio_array_to_bytes(array: ArrayView2<f32>, dtype: DType) -> Result<Vec<u8>, Box<dyn Error>> {
    match dtype {
        DType::Fp32 => {
            let mut result = Vec::with_capacity(array.len() * 4);
            for &value in array.iter() {
                result.extend_from_slice(&value.to_le_bytes());
            }
            Ok(result)
        }
        _ => Err("Unsupported audio data type".into()),
    }
}

fn parse_tensor_bytes(bytes: &[u8], dtype: DType) -> Result<Vec<f32>, Box<dyn Error>> {
    match dtype {
        DType::Fp32 => {
            let mut result = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result.push(value);
            }
            Ok(result)
        }
        _ => Err("Unsupported tensor data type".into()),
    }
}

fn tensor_array_to_bytes(
    array: ArrayView<f32, ndarray::IxDyn>,
    dtype: DType,
) -> Result<Vec<u8>, Box<dyn Error>> {
    match dtype {
        DType::Fp32 => {
            let mut result = Vec::with_capacity(array.len() * 4);
            for &value in array.iter() {
                result.extend_from_slice(&value.to_le_bytes());
            }
            Ok(result)
        }
        _ => Err("Unsupported tensor data type".into()),
    }
}

impl Serializer for OnnxSerializer {
    fn serialize(
        &self,
        schema: &ValueSchema,
        value: ProtoValue,
    ) -> Result<OrtValue, Box<dyn Error>> {
        match schema.value_type.as_ref().ok_or("Missing value type")? {
            ValueType::JointPositions(ref joint_positions_schema) => match value.value {
                Some(EnumValue::JointPositions(values)) => {
                    self.serialize_joint_positions(joint_positions_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::JointVelocities(ref joint_velocities_schema) => match value.value {
                Some(EnumValue::JointVelocities(values)) => {
                    self.serialize_joint_velocities(joint_velocities_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::JointTorques(ref joint_torques_schema) => match value.value {
                Some(EnumValue::JointTorques(values)) => {
                    self.serialize_joint_torques(joint_torques_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::JointCommands(ref joint_commands_schema) => match value.value {
                Some(EnumValue::JointCommands(values)) => {
                    self.serialize_joint_commands(joint_commands_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::CameraFrame(ref camera_frame_schema) => match value.value {
                Some(EnumValue::CameraFrame(values)) => {
                    self.serialize_camera_frame(camera_frame_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::AudioFrame(ref audio_frame_schema) => match value.value {
                Some(EnumValue::AudioFrame(values)) => {
                    self.serialize_audio_frame(audio_frame_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::Imu(ref imu_schema) => match value.value {
                Some(EnumValue::Imu(values)) => self.serialize_imu(imu_schema, values),
                _ => Err("Unsupported value type".into()),
            },
            ValueType::Timestamp(ref timestamp_schema) => match value.value {
                Some(EnumValue::Timestamp(values)) => {
                    self.serialize_timestamp(timestamp_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::VectorCommand(ref vector_command_schema) => match value.value {
                Some(EnumValue::VectorCommand(values)) => {
                    self.serialize_vector_command(vector_command_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
            ValueType::StateTensor(ref state_tensor_schema) => match value.value {
                Some(EnumValue::StateTensor(values)) => {
                    self.serialize_state_tensor(state_tensor_schema, values)
                }
                _ => Err("Unsupported value type".into()),
            },
        }
    }

    fn deserialize(
        &self,
        schema: &ValueSchema,
        value: OrtValue,
    ) -> Result<ProtoValue, Box<dyn Error>> {
        match schema.value_type.as_ref().ok_or("Missing value type")? {
            ValueType::JointPositions(ref joint_positions_schema) => {
                let positions = self.deserialize_joint_positions(joint_positions_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::JointPositions(positions)),
                })
            }
            ValueType::JointVelocities(ref joint_velocities_schema) => {
                let velocities =
                    self.deserialize_joint_velocities(joint_velocities_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::JointVelocities(velocities)),
                })
            }
            ValueType::JointTorques(ref joint_torques_schema) => {
                let torques = self.deserialize_joint_torques(joint_torques_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::JointTorques(torques)),
                })
            }
            ValueType::JointCommands(ref joint_commands_schema) => {
                let commands = self.deserialize_joint_commands(joint_commands_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::JointCommands(commands)),
                })
            }
            ValueType::CameraFrame(ref camera_frame_schema) => {
                let frame = self.deserialize_camera_frame(camera_frame_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::CameraFrame(frame)),
                })
            }
            ValueType::AudioFrame(ref audio_frame_schema) => {
                let frame = self.deserialize_audio_frame(audio_frame_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::AudioFrame(frame)),
                })
            }
            ValueType::Imu(ref imu_schema) => {
                let imu = self.deserialize_imu(imu_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::Imu(imu)),
                })
            }
            ValueType::Timestamp(ref timestamp_schema) => {
                let timestamp = self.deserialize_timestamp(timestamp_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::Timestamp(timestamp)),
                })
            }
            ValueType::VectorCommand(ref vector_command_schema) => {
                let command = self.deserialize_vector_command(vector_command_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::VectorCommand(command)),
                })
            }
            ValueType::StateTensor(ref state_tensor_schema) => {
                let tensor = self.deserialize_state_tensor(state_tensor_schema, value)?;
                Ok(ProtoValue {
                    value_name: schema.value_name.clone(),
                    value: Some(EnumValue::StateTensor(tensor)),
                })
            }
        }
    }
}

fn calculate_value_size(schema: &ValueSchema) -> Result<usize, Box<dyn Error>> {
    match schema.value_type.as_ref().ok_or("Missing value type")? {
        ValueType::JointPositions(s) => Ok(s.joint_names.len()),
        ValueType::JointVelocities(s) => Ok(s.joint_names.len()),
        ValueType::JointTorques(s) => Ok(s.joint_names.len()),
        ValueType::JointCommands(s) => Ok(s.joint_names.len() * 5), // 5 values per joint
        ValueType::CameraFrame(s) => Ok((s.channels * s.height * s.width) as usize),
        ValueType::AudioFrame(s) => Ok((s.channels * s.sample_rate) as usize),
        ValueType::Imu(s) => {
            let mut size = 0;
            if s.use_accelerometer {
                size += 3;
            }
            if s.use_gyroscope {
                size += 3;
            }
            if s.use_magnetometer {
                size += 3;
            }
            Ok(size)
        }
        ValueType::Timestamp(_) => Ok(1),
        ValueType::VectorCommand(s) => Ok(s.dimensions as usize),
        ValueType::StateTensor(s) => Ok(s.shape.iter().product::<i32>() as usize),
    }
}

pub struct OnnxMultiSerializer {
    serializers: Vec<OnnxSerializer>,
}

impl OnnxMultiSerializer {
    pub fn new(schema: ProtoIOSchema) -> Self {
        Self {
            serializers: schema
                .values
                .into_iter()
                .map(|s| OnnxSerializer::new(s))
                .collect(),
        }
    }

    pub fn serialize_io(&self, io: ProtoIO) -> Result<OrtValue, Box<dyn Error>> {
        if io.values.len() != self.serializers.len() {
            return Err("Number of values does not match schema".into());
        }

        // Serialize each value according to its schema and concatenate the results
        let mut all_values: Vec<f32> = Vec::new();
        for (value, serializer) in io.values.iter().zip(self.serializers.iter()) {
            let tensor = serializer.serialize(&serializer.schema, value.clone())?;
            let array = tensor.try_extract_tensor::<f32>()?;
            let array_1d = array
                .as_standard_layout()
                .into_dimensionality::<ndarray::Ix1>()?;
            all_values.extend(array_1d.iter().copied());
        }

        // Convert to OrtValue
        Tensor::from_array(Array1::from_vec(all_values))
            .map(|tensor| tensor.into_dyn())
            .map_err(|e| Box::new(e) as Box<dyn Error>)
    }

    pub fn deserialize_io(&self, values: Vec<OrtValue>) -> Result<ProtoIO, Box<dyn Error>> {
        // Check if number of values matches number of serializers
        if values.len() != self.serializers.len() {
            return Err(format!(
                "Number of values ({}) does not match number of serializers ({})",
                values.len(),
                self.serializers.len()
            )
            .into());
        }

        // Deserialize each value using its corresponding serializer
        let proto_values = self
            .serializers
            .iter()
            .zip(values.into_iter())
            .map(|(serializer, value)| serializer.deserialize(&serializer.schema, value))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ProtoIO {
            values: proto_values,
        })
    }

    pub fn names(&self) -> Vec<String> {
        self.serializers
            .iter()
            .map(|s| s.schema.value_name.clone())
            .collect()
    }

    pub fn assign_names(
        &self,
        values: Vec<OrtValue>,
    ) -> Result<std::collections::HashMap<String, OrtValue>, Box<dyn Error>> {
        if values.len() != self.serializers.len() {
            return Err(format!(
                "Expected {} values, got {}",
                self.serializers.len(),
                values.len()
            )
            .into());
        }

        Ok(self
            .serializers
            .iter()
            .map(|s| s.schema.value_name.clone())
            .zip(values)
            .collect())
    }
}
