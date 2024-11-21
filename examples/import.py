"""Tests for model inference functionality."""

from pathlib import Path

import numpy as np
import torch
from export import ModelConfig, SimpleModel

from kinfer.export.pytorch import export_to_onnx
from kinfer.inference.python import ONNXModel


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
    assert metadata["hidden_size"] == "64"
    assert metadata["num_layers"] == "2"


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


def main() -> None:
    """Run inference example."""
    print("Creating and exporting model...")

    # Create and export model
    config = ModelConfig(hidden_size=64, num_layers=2)
    model = SimpleModel(config)

    # Create example input
    input_tensor = torch.randn(1, 10)

    # Export model to ONNX
    save_path = "simple_model.onnx"
    export_to_onnx(model=model, input_tensors=input_tensor, config=config, save_path=save_path)
    print(f"Model exported to {save_path}")

    # Load model for inference
    print("\nLoading model for inference...")
    onnx_model = ONNXModel(save_path)

    # Print model information
    print("\nModel metadata:")
    for key, value in onnx_model.get_metadata().items():
        print(f"  {key}: {value}")

    print("\nInput details:")
    for detail in onnx_model.get_input_details():
        print(f"  {detail}")

    print("\nOutput details:")
    for detail in onnx_model.get_output_details():
        print(f"  {detail}")

    # Run inference
    print("\nRunning inference...")
    input_data = np.random.randn(1, 10).astype(np.float32)

    # Method 1: Direct numpy array input
    output1 = onnx_model(input_data)
    print("\nMethod 1 - Direct numpy array input:")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {output1.shape if isinstance(output1, np.ndarray) else 'N/A'}")
    print(f"  Output value: {output1}")

    # Method 2: Dictionary input
    input_name = onnx_model.get_input_details()[0]["name"]
    output2 = onnx_model({input_name: input_data})
    assert isinstance(output2, dict)
    print("\nMethod 2 - Dictionary input:")
    print(f"  Output keys: {list(output2.keys())}")
    print(f"  Output shapes: {[arr.shape for arr in output2.values() if isinstance(arr, np.ndarray)]}")

    # Method 3: List input
    output3 = onnx_model([input_data])
    print("\nMethod 3 - List input:")
    print(f"  Output length: {len(output3)}")
    print(f"  Output shapes: {[arr.shape for arr in output3 if isinstance(arr, np.ndarray)]}")


if __name__ == "__main__":
    main()
