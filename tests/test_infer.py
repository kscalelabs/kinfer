"""Tests for model inference functionality."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from kinfer import proto as P
from kinfer.export.pytorch import export_model
from kinfer.inference.python import ONNXModel


@dataclass
class ModelConfig:
    in_features: int = 10
    hidden_size: int = 64
    num_layers: int = 2


class SimpleModel(torch.nn.Module):
    """A simple neural network model for demonstration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []

        in_features = config.in_features
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
    jit_model = torch.jit.script(model)

    save_path = str(tmp_path / "test_model.onnx")
    exported_model = export_model(
        model=jit_model,
        schema=P.ModelSchema(
            input_schema=P.IOSchema(
                values=[
                    P.ValueSchema(
                        value_name="x",
                        state_tensor=P.StateTensorSchema(
                            shape=[1, config.in_features],
                            dtype=P.DType.FP32,
                        ),
                    ),
                ],
            ),
            output_schema=P.IOSchema(
                values=[
                    P.ValueSchema(
                        value_name="output",
                        state_tensor=P.StateTensorSchema(
                            shape=[1, 1],
                            dtype=P.DType.FP32,
                        ),
                    ),
                ],
            ),
        ),
    )
    onnx.save_model(exported_model, save_path)

    return save_path


def test_model_loading(model_path: str) -> None:
    """Test basic model loading functionality."""
    model = ONNXModel(model_path)
    assert model is not None


def test_model_inference(model_path: str) -> None:
    """Test model inference with different input formats."""
    model = ONNXModel(model_path)

    inputs = P.IO(
        values=[
            P.Value(
                state_tensor=P.StateTensorValue(data=np.random.randn(1, 10).astype(np.float32).tobytes()),
            ),
        ],
    )
    outputs = model(inputs)
    assert isinstance(outputs, P.IO)
