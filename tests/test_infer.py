"""Tests for model inference functionality."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from kinfer.export.pytorch import export_to_onnx
from kinfer.inference.python import ONNXModel


@dataclass
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2


class SimpleModel(torch.nn.Module):
    """A simple neural network model for demonstration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []
        in_features = 10

        for _ in range(config.num_layers):
            layers.extend([torch.nn.Linear(in_features, config.hidden_size), torch.nn.ReLU()])
            in_features = config.hidden_size

        layers.append(torch.nn.Linear(config.hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def model_path(tmp_path: Path) -> str:
    """Create and export a test model."""
    # Create and export model
    config = ModelConfig()
    model = SimpleModel(config)

    save_path = str(tmp_path / "test_model.onnx")
    export_to_onnx(model=model, input_tensors=torch.randn(1, 10), config=config, save_path=save_path)

    return save_path


def test_model_loading(model_path: str) -> None:
    """Test basic model loading functionality."""
    # Test with default config
    model = ONNXModel(model_path)
    assert model is not None

    model = ONNXModel(model_path)
    assert model is not None


def test_model_metadata(model_path: str) -> None:
    """Test model metadata extraction."""
    model = ONNXModel(model_path)
    metadata = model.get_metadata()

    # Check if config parameters are in metadata
    assert "hidden_size" in metadata
    assert "num_layers" in metadata
    assert metadata["hidden_size"] == 64
    assert metadata["num_layers"] == 2


def test_model_inference(model_path: str) -> None:
    """Test model inference with different input formats."""
    model = ONNXModel(model_path)

    # Test with numpy array
    input_data = np.random.randn(1, 10).astype(np.float32)
    output = model(input_data)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1)

    # Test with dictionary input
    input_name = model.get_input_details()[0]["name"]
    output = model({input_name: input_data})
    assert isinstance(output, dict)

    # Test with list input
    output = model([input_data])
    assert isinstance(output, list)


def test_model_details(model_path: str) -> None:
    """Test input/output detail extraction."""
    model = ONNXModel(model_path)

    # Check input details
    input_details = model.get_input_details()
    assert len(input_details) == 1
    assert input_details[0]["shape"] == [1, 10]

    # Check output details
    output_details = model.get_output_details()
    assert len(output_details) == 1
    assert output_details[0]["shape"] == [1, 1]


def test_comprehensive_model_workflow(tmp_path: Path) -> None:
    """Test complete model workflow including export, loading and inference."""
    # Create and export model
    config = ModelConfig(hidden_size=64, num_layers=2)
    model = SimpleModel(config)
    input_tensor = torch.randn(1, 10)

    save_path = str(tmp_path / "test_model.onnx")
    export_to_onnx(model=model, input_tensors=input_tensor, config=config, save_path=save_path)

    # Load model for inference
    onnx_model = ONNXModel(save_path)

    # Test metadata
    metadata = onnx_model.get_metadata()
    assert "hidden_size" in metadata
    assert "num_layers" in metadata
    assert metadata["hidden_size"] == 64
    assert metadata["num_layers"] == 2

    # Test input/output details
    input_details = onnx_model.get_input_details()
    assert len(input_details) == 1
    assert input_details[0]["shape"] == [1, 10]

    output_details = onnx_model.get_output_details()
    assert len(output_details) == 1
    assert output_details[0]["shape"] == [1, 1]

    # Test inference with different input methods
    input_data = np.random.randn(1, 10).astype(np.float32)

    # Method 1: Direct numpy array input
    output1 = onnx_model(input_data)
    assert isinstance(output1, np.ndarray)
    assert output1.shape == (1, 1)

    # Method 2: Dictionary input
    input_name = onnx_model.get_input_details()[0]["name"]
    output2 = onnx_model({input_name: input_data})
    assert isinstance(output2, dict)
    assert len(output2) == 1
    assert list(output2.values())[0].shape == (1, 1)

    # Method 3: List input
    output3 = onnx_model([input_data])
    assert isinstance(output3, list)
    assert len(output3) == 1
    assert output3[0].shape == (1, 1)


def test_export_with_given_input(tmp_path: Path) -> None:
    """Test model export with explicitly provided input tensor."""
    config = ModelConfig()
    model = SimpleModel(config)

    # Create specific input tensor
    input_tensor = torch.randn(1, 10)

    save_path = str(tmp_path / "explicit_input_model.onnx")
    session = export_to_onnx(model=model, input_tensors=input_tensor, config=config, save_path=save_path)

    # Verify input shape matches what we provided
    inputs = session.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].shape == [1, 10]


def test_export_with_inferred_input(tmp_path: Path) -> None:
    """Test model export with automatically inferred input tensor."""
    config = ModelConfig()
    model = SimpleModel(config)

    save_path = str(tmp_path / "inferred_input_model.onnx")
    session = export_to_onnx(
        model=model,
        input_tensors=None,
        config=config,
        save_path=save_path,  # Let it infer the input
    )

    # Verify input shape was correctly inferred
    inputs = session.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].shape == [1, 10]  # Should match the in_features=10 from SimpleModel
