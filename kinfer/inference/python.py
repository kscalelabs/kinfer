"""ONNX model inference utilities for Python."""

import base64
from pathlib import Path

import onnx
import onnxruntime as ort

from kinfer import proto as P
from kinfer.export.pytorch import KINFER_METADATA_KEY
from kinfer.serialize.numpy import NumpyMultiSerializer


class ONNXModel:
    """Wrapper for ONNX model inference."""

    def __init__(self, model_path: str | Path) -> None:
        """Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path

        # Load model and create inference session
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(model_path)
        self.attached_metadata: dict[str, str] = {}

        # Extract metadata and attempt to parse JSON values
        for prop in self.model.metadata_props:
            if prop.key == KINFER_METADATA_KEY:
                schema_bytes = base64.b64decode(prop.value)
                schema = P.ModelSchema()
                schema.ParseFromString(schema_bytes)
            else:
                self.attached_metadata[prop.key] = prop.value

        # Extract input and output schemas from metadata
        self._schema = schema

        # Create serializers for input and output.
        self._input_serializer = NumpyMultiSerializer(self._schema.input_schema)
        self._output_serializer = NumpyMultiSerializer(self._schema.output_schema)

    def __call__(self, inputs: P.IO) -> P.IO:
        """Run inference on input data.

        Args:
            inputs: Input data, matching the input schema.

        Returns:
            Model outputs, matching the output schema.
        """
        inputs_np = self._input_serializer.serialize_io(inputs, as_dict=True)
        outputs_np = self.session.run(None, inputs_np)
        outputs = self._output_serializer.deserialize_io(outputs_np)
        return outputs

    @property
    def input_schema(self) -> P.IOSchema:
        """Get the input schema."""
        return self._schema.input_schema

    @property
    def output_schema(self) -> P.IOSchema:
        """Get the output schema."""
        return self._schema.output_schema
