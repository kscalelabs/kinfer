"""ONNX model inference utilities for Python."""

import json
import logging
from typing import Any, Dict, List, Union

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]


class ONNXModel:
    """Wrapper for ONNX model inference."""

    def __init__(
        self,
        model_path: str,
    ) -> None:
        """Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
            config: Optional inference configuration
        """
        self.model_path = model_path

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
        self, inputs: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]:
        """Run inference on input data.

        Args:
            inputs: Input data as numpy array, dictionary of arrays, or list of arrays

        Returns:
            Model outputs in the same format as inputs
        """
        # Convert single array to dict
        if isinstance(inputs, np.ndarray):
            input_dict = {self.input_details[0]["name"]: inputs}
        # Convert list to dict
        elif isinstance(inputs, list):
            input_dict = {detail["name"]: arr for detail, arr in zip(self.input_details, inputs)}
        else:
            input_dict = inputs

        # Run inference - pass None to output_names param to get all outputs
        outputs = self.session.run(None, input_dict)

        # Convert output format to match input
        if isinstance(inputs, np.ndarray):
            return outputs[0]
        elif isinstance(inputs, list):
            return outputs
        else:
            return {detail["name"]: arr for detail, arr in zip(self.output_details, outputs)}

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
