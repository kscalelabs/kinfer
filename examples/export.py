#!/usr/bin/env python
"""Example script demonstrating model export functionality."""

import logging
from dataclasses import dataclass

import torch
from torch import nn

from kinfer.export.pytorch import export_model


@dataclass
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2


class SimpleModel(nn.Module):
    """A simple neural network model for demonstration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []
        in_features = 10  # Example input size

        for _ in range(config.num_layers):
            layers.extend([nn.Linear(in_features, config.hidden_size), nn.ReLU()])
            in_features = config.hidden_size

        layers.append(nn.Linear(config.hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = ModelConfig()
    model = SimpleModel(config)

    # Create example input
    # batch_size = 1
    # input_tensor = torch.randn(batch_size, 10)

    # Export model to ONNX
    session = export_model(model=model, input_tensors=None, config=config, save_path="simple_model.onnx")

    logger.info("Model exported successfully!")
    inputs = [{"name": node.name, "shape": node.shape, "type": node.type} for node in session.get_inputs()]
    outputs = [{"name": node.name, "shape": node.shape, "type": node.type} for node in session.get_outputs()]
    logger.info("Model inputs: %s", inputs)
    logger.info("Model outputs: %s", outputs)


if __name__ == "__main__":
    main()
