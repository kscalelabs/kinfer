"""ONNX model inference utilities for Python."""

import base64
from pathlib import Path

import onnx
import onnxruntime as ort

from kinfer import proto as K
from kinfer.export.pytorch import KINFER_METADATA_KEY
from kinfer.inference.base import KModel
from kinfer.serialize.numpy import NumpyMultiSerializer


def _read_schema(model: onnx.ModelProto) -> K.ModelSchema:
    for prop in model.metadata_props:
        if prop.key == KINFER_METADATA_KEY:
            try:
                schema_bytes = base64.b64decode(prop.value)
                schema = K.ModelSchema()
                schema.ParseFromString(schema_bytes)
                return schema
            except Exception as e:
                raise ValueError("Failed to parse kinfer_metadata value") from e
        else:
            raise ValueError(f"Found arbitrary metadata key {prop.key}")

    raise ValueError(f"{KINFER_METADATA_KEY} not found in model metadata")


class ONNXModel(KModel):
    """Wrapper for ONNX model inference."""

    def __init__(self: "ONNXModel", model_path: str | Path) -> None:
        """Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path

        # Load model and create inference session
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(model_path)
        self._schema = _read_schema(self.model)

        # Create serializers for input and output.
        self._input_serializer = NumpyMultiSerializer(self._schema.input_schema)
        self._output_serializer = NumpyMultiSerializer(self._schema.output_schema)

    def get_schema(self) -> K.ModelSchema:
        return self._schema

    def __call__(self, inputs: K.IO) -> K.IO:
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
