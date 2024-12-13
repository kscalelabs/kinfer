"""ONNX model inference utilities for Python."""

import json
import logging
from typing import Any, Dict, List, Union

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from kinfer.serialize.pytorch import PyTorchInputSerializer, PyTorchOutputSerializer
from kinfer.protos.kinfer_pb2 import InputSchema, OutputSchema, Input, Output


class ONNXModel:
    """Wrapper for ONNX model inference."""

    def __init__(
        self,
        model_path: str,
        input_schema: InputSchema,
        output_schema: OutputSchema,
    ) -> None:
        """Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
            config: Optional inference configuration
        """
        self.model_path = model_path
        self.input_serializer = PyTorchInputSerializer(schema=input_schema)
        self.output_serializer = PyTorchOutputSerializer(schema=output_schema)

        # Load model and create inference session
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(
            model_path,
        )

        # Extract metadata and attempt to parse JSON values
        self.metadata = {}
        self.attached_metadata = {}
        for prop in self.model.metadata_props:
            if prop.key == "kinfer_metadata":
                try:
                    self.metadata = json.loads(prop.value)
                except json.JSONDecodeError:
                    logging.warning(
                        "Failed to parse kinfer_metadata value with JSON parser. Saving as string: %s",
                        prop.value,
                    )
                    self.metadata = prop.value

            self.attached_metadata[prop.key] = prop.value

        # Get input and output details
        self.input_details = [{"name": x.name, "shape": x.shape, "type": x.type} for x in self.session.get_inputs()]
        self.output_details = [{"name": x.name, "shape": x.shape, "type": x.type} for x in self.session.get_outputs()]

    def __call__(
        self, input: Input
    ) -> Output:
        """Run inference on input data.

        Args:
            inputs: Repeated Value objects

        Returns:
            Repeated Value objects
        """
        serialized_inputs = self.input_serializer.serialize(input)

        breakpoint()
        # Run inference - pass None to output_names param to get all outputs
        outputs = self.session.run(None, serialized_inputs)

        return self.output_serializer.deserialize(outputs)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Directly passes inputs to the model and returns outputs.

        Args:
            inputs: Input data as dictionary of arrays

        Returns:
            Model outputs as dictionary of arrays
        """
        return self.session.run(None, inputs)

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary of metadata key-value pairs
        """
        return self.metadata

    def get_input_details(self) -> List[Dict[str, Any]]:
        """Get input tensor details.

        Returns:
            List of dictionaries containing input tensor information
        """
        return self.input_details

    def get_output_details(self) -> List[Dict[str, Any]]:
        """Get output tensor details.

        Returns:
            List of dictionaries containing output tensor information
        """
        return self.output_details
