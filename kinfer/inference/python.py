"""ONNX model inference utilities for Python."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from kinfer import protos as P
from kinfer.export.pytorch import KINFER_METADATA_KEY


class ONNXModel:
    """Wrapper for ONNX model inference."""

    def __init__(self, model_path: str | Path) -> None:
        """Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
            config: Optional inference configuration
        """
        self.model_path = model_path

        # Load model and create inference session
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(model_path)

        # Extract metadata and attempt to parse JSON values
        for prop in self.model.metadata_props:
            if prop.key == KINFER_METADATA_KEY:
                try:
                    metadata = P.ModelSchema.FromString(prop.value.encode("utf-8"))
                except Exception as e:
                    raise ValueError("Failed to parse kinfer_metadata value") from e
                break
            else:
                self.attached_metadata[prop.key] = prop.value
        else:
            raise ValueError("kinfer_metadata not found in model metadata")

        self._input_schema = metadata.input_schema
        self._output_schema = metadata.output_schema

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on input data.

        Args:
            inputs: Input data, matching the input schema.

        Returns:
            Model outputs, matching the output schema.
        """
        # Run inference - pass None to output_names param to get all outputs
        outputs = self.session.run(None, inputs)
        return outputs

    @property
    def input_schema(self) -> P.InputSchema:
        """Get the input schema."""
        return self._input_schema

    @property
    def output_schema(self) -> P.OutputSchema:
        """Get the output schema."""
        return self._output_schema
