pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/proto/kinfer.proto.rs"));
}

pub use proto::{
    AudioFrameSchema, AudioFrameValue, CameraFrameSchema, CameraFrameValue, DType, ImuAccelerometerValue,
    ImuGyroscopeValue, ImuMagnetometerValue, ImuSchema, ImuValue, JointCommandValue,
    JointCommandsSchema, JointCommandsValue, JointPositionUnit, JointPositionValue,
    JointPositionsSchema, JointPositionsValue, JointTorqueUnit, JointTorqueValue, JointTorquesSchema,
    JointTorquesValue, JointVelocitiesSchema, JointVelocitiesValue, JointVelocityUnit,
    JointVelocityValue, StateTensorSchema, StateTensorValue, TimestampSchema, TimestampValue,
    VectorCommandSchema, VectorCommandValue, Value as ProtoValue, ValueSchema, Io as ProtoIO, IoSchema as ProtoIOSchema,
    ModelSchema
};
