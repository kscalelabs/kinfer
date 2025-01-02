pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/proto/kinfer.proto.rs"));
}

pub use proto::{
    AudioFrameSchema, AudioFrameValue, CameraFrameSchema, CameraFrameValue, DType,
    ImuAccelerometerValue, ImuGyroscopeValue, ImuMagnetometerValue, ImuSchema, ImuValue,
    Io as ProtoIO, IoSchema as ProtoIOSchema, JointCommandValue, JointCommandsSchema,
    JointCommandsValue, JointPositionUnit, JointPositionValue, JointPositionsSchema,
    JointPositionsValue, JointTorqueUnit, JointTorqueValue, JointTorquesSchema, JointTorquesValue,
    JointVelocitiesSchema, JointVelocitiesValue, JointVelocityUnit, JointVelocityValue,
    ModelSchema, StateTensorSchema, StateTensorValue, TimestampSchema, TimestampValue,
    Value as ProtoValue, ValueSchema, VectorCommandSchema, VectorCommandValue,
};
