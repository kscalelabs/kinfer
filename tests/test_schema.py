"""Tests for model schema functionality."""

from pathlib import Path

import onnx
import pytest
import torch

from kinfer import proto as K
from kinfer.export.pytorch import KINFER_METADATA_KEY, export_model


@pytest.fixture
def complex_schema() -> K.ModelSchema:
    """Create a complex model schema for testing."""
    return K.ModelSchema(
        input_schema=K.IOSchema(
            values=[
                K.ValueSchema(
                    value_name="joint_positions",
                    joint_positions=K.JointPositionsSchema(
                        unit=K.JointPositionUnit.DEGREES,
                        joint_names=["joint1", "joint2", "joint3"],
                    ),
                ),
                K.ValueSchema(
                    value_name="joint_velocities",
                    joint_velocities=K.JointVelocitiesSchema(
                        unit=K.JointVelocityUnit.DEGREES_PER_SECOND,
                        joint_names=["joint1", "joint2", "joint3"],
                    ),
                ),
                K.ValueSchema(
                    value_name="joint_torques",
                    joint_torques=K.JointTorquesSchema(
                        unit=K.JointTorqueUnit.NEWTON_METERS,
                        joint_names=["joint1", "joint2", "joint3"],
                    ),
                ),
                K.ValueSchema(
                    value_name="camera_frame",
                    camera_frame=K.CameraFrameSchema(
                        width=64,
                        height=64,
                        channels=3,
                    ),
                ),
                K.ValueSchema(
                    value_name="audio_frame",
                    audio_frame=K.AudioFrameSchema(
                        channels=1,
                        sample_rate=16000,
                        dtype=K.DType.FP32,
                    ),
                ),
                K.ValueSchema(
                    value_name="imu",
                    imu=K.ImuSchema(
                        use_accelerometer=True,
                        use_gyroscope=True,
                        use_magnetometer=True,
                    ),
                ),
                K.ValueSchema(
                    value_name="timestamp",
                    timestamp=K.TimestampSchema(),
                ),
                K.ValueSchema(
                    value_name="vector_command",
                    vector_command=K.VectorCommandSchema(
                        dimensions=3,
                    ),
                ),
                K.ValueSchema(
                    value_name="state_tensor",
                    state_tensor=K.StateTensorSchema(
                        shape=[1, 10],
                        dtype=K.DType.FP32,
                    ),
                ),
            ],
        ),
        output_schema=K.IOSchema(
            values=[
                K.ValueSchema(
                    value_name="joint_commands",
                    joint_commands=K.JointCommandsSchema(
                        joint_names=["joint1", "joint2", "joint3"],
                        torque_unit=K.JointTorqueUnit.NEWTON_METERS,
                        velocity_unit=K.JointVelocityUnit.RADIANS_PER_SECOND,
                        position_unit=K.JointPositionUnit.RADIANS,
                    ),
                ),
            ],
        ),
    )


class DummyModel(torch.nn.Module):
    """A dummy model for testing schema persistence."""

    def __init__(self: "DummyModel") -> None:
        super().__init__()
        # Joint positions, velocities, torques processing
        self.joint_linear = torch.nn.Linear(9, 15)  # 3 joints × 3 types

        # Camera frame processing
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Audio processing
        self.audio_linear = torch.nn.Linear(16000, 15)

        # IMU processing
        self.imu_linear = torch.nn.Linear(9, 15)  # 3 values × 3 sensors

        # Timestamp processing
        self.time_embedding = torch.nn.Linear(1, 15)

        # Vector command processing
        self.vec_linear = torch.nn.Linear(3, 15)

        # State tensor processing
        self.state_linear = torch.nn.Linear(10, 15)

        # Final processing
        self.final = torch.nn.Linear(7 * 15, 15)  # 7 inputs × 15 features

    def forward(
        self: "DummyModel",
        joint_positions: torch.Tensor,  # [3]
        joint_velocities: torch.Tensor,  # [3]
        joint_torques: torch.Tensor,  # [3]
        camera_frame: torch.Tensor,  # [3, 64, 64]
        audio_frame: torch.Tensor,  # [16000]
        imu: torch.Tensor,  # [9]
        timestamp: torch.Tensor,  # [1]
        vector_command: torch.Tensor,  # [3]
        state_tensor: torch.Tensor,  # [1, 10]
    ) -> torch.Tensor:
        # Process joints
        joints = torch.cat([joint_positions, joint_velocities, joint_torques])
        joint_features = self.joint_linear(joints)  # [15]

        # Process camera
        camera_features = self.conv(camera_frame.unsqueeze(0))  # [1, 16, H, W]
        camera_features = self.pool(camera_features)  # [1, 16, 1, 1]
        camera_features = camera_features.squeeze(-1).squeeze(-1)  # [1, 16]
        camera_features = camera_features.squeeze(0)  # [16]
        camera_features = camera_features[:15]  # [15] to match other feature dimensions

        # Process audio
        audio_features = self.audio_linear(audio_frame).squeeze(0)  # [15]

        # Process IMU
        imu_features = self.imu_linear(imu.flatten())  # [15]

        # Process timestamp
        time_features = self.time_embedding(timestamp)  # [15]

        # Process vector command
        vec_features = self.vec_linear(vector_command)  # [15]

        # Process state tensor
        state_features = self.state_linear(state_tensor.squeeze(0))  # [15]

        print(f"joint_features: {joint_features.shape}")
        print(f"camera_features: {camera_features.shape}")
        print(f"audio_features: {audio_features.shape}")
        print(f"imu_features: {imu_features.shape}")
        print(f"time_features: {time_features.shape}")
        print(f"vec_features: {vec_features.shape}")
        print(f"state_features: {state_features.shape}")

        # Combine all features
        combined = torch.cat(
            [
                joint_features,
                camera_features,
                audio_features,
                imu_features,
                time_features,
                vec_features,
                state_features,
            ]
        )

        # Final processing to get joint commands
        output = self.final(combined)
        return output.reshape(3, 5)  # [3, 5] for 3 joints × (pos, vel, torque, kp, kd)


def test_schema_persistence(tmp_path: Path, complex_schema: K.ModelSchema) -> None:
    """Test that schema is correctly persisted in model metadata."""
    model = DummyModel()
    jit_model = torch.jit.script(model)

    # Export model with schema
    exported_model = export_model(
        model=jit_model,
        schema=complex_schema,
    )

    # Save and reload model
    save_path = str(tmp_path / "test_model.onnx")
    onnx.save_model(exported_model, save_path)
    loaded_model = onnx.load(save_path)

    # Get schema from metadata
    metadata_props = {prop.key: prop.value for prop in loaded_model.metadata_props}
    assert KINFER_METADATA_KEY in metadata_props

    # Load schema from model and verify it matches original
    from kinfer.inference.python import ONNXModel

    model = ONNXModel(save_path)
    loaded_schema = model._schema

    # Verify input schema
    assert len(loaded_schema.input_schema.values) == len(complex_schema.input_schema.values)
    for orig_val, loaded_val in zip(complex_schema.input_schema.values, loaded_schema.input_schema.values):
        assert orig_val.value_name == loaded_val.value_name
        assert orig_val.WhichOneof("value_type") == loaded_val.WhichOneof("value_type")

    # Verify output schema
    assert len(loaded_schema.output_schema.values) == len(complex_schema.output_schema.values)
    for orig_val, loaded_val in zip(complex_schema.output_schema.values, loaded_schema.output_schema.values):
        assert orig_val.value_name == loaded_val.value_name
        assert orig_val.WhichOneof("value_type") == loaded_val.WhichOneof("value_type")
