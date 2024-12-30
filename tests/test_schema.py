"""Tests for model schema functionality."""

from pathlib import Path

import onnx
import pytest
import torch

from kinfer import proto as P
from kinfer.export.pytorch import KINFER_METADATA_KEY, export_model


@pytest.fixture
def complex_schema() -> P.ModelSchema:
    """Create a complex model schema for testing."""
    return P.ModelSchema(
        input_schema=P.IOSchema(
            values=[
                P.ValueSchema(
                    value_name="joint_positions",
                    joint_positions=P.JointPositionsSchema(
                        unit=P.JointPositionUnit.DEGREES,
                        joint_names=["joint1", "joint2", "joint3"],
                    ),
                ),
                P.ValueSchema(
                    value_name="camera",
                    camera_frame=P.CameraFrameSchema(
                        width=640,
                        height=480,
                        channels=3,
                    ),
                ),
            ],
        ),
        output_schema=P.IOSchema(
            values=[
                P.ValueSchema(
                    value_name="joint_commands",
                    joint_commands=P.JointCommandsSchema(
                        joint_names=["joint1", "joint2", "joint3"],
                        torque_unit=P.JointTorqueUnit.NEWTON_METERS,
                        velocity_unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
                        position_unit=P.JointPositionUnit.RADIANS,
                    ),
                ),
            ],
        ),
    )


class DummyModel(torch.nn.Module):
    """A dummy model for testing schema persistence."""

    def __init__(self: "DummyModel") -> None:
        super().__init__()
        self.joint_linear = torch.nn.Linear(3, 15)  # 3 joints
        self.camera_conv = torch.nn.Conv2d(3, 1, kernel_size=3)  # RGB image

    def forward(
        self: "DummyModel",
        joint_positions: torch.Tensor,
        camera: torch.Tensor,
    ) -> torch.Tensor:
        joint_features = self.joint_linear(joint_positions)
        camera_features = self.camera_conv(camera).flatten(1)
        return torch.cat([joint_features, camera_features], dim=1)


def test_schema_persistence(tmp_path: Path, complex_schema: P.ModelSchema) -> None:
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


def test_schema_validation(tmp_path: Path, complex_schema: P.ModelSchema) -> None:
    """Test schema validation during model loading."""
    model = DummyModel()
    jit_model = torch.jit.script(model)

    # Export model with schema
    exported_model = export_model(
        model=jit_model,
        schema=complex_schema,
    )

    # Save model
    save_path = str(tmp_path / "test_model.onnx")
    onnx.save_model(exported_model, save_path)

    # Corrupt the schema metadata
    loaded_model = onnx.load(save_path)
    for prop in loaded_model.metadata_props:
        if prop.key == KINFER_METADATA_KEY:
            prop.value = "invalid_base64_data"
    onnx.save_model(loaded_model, save_path)

    # Verify loading raises error
    from kinfer.inference.python import ONNXModel
    with pytest.raises(ValueError, match="Failed to decode schema"):
        ONNXModel(save_path)
