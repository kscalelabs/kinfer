"""PyTorch model export utilities."""

import inspect
from io import BytesIO
from typing import Sequence

import onnx
import onnxruntime as ort
import torch
from torch import Tensor

from kinfer import protos as P
from kinfer.serialize.pytorch import PyTorchMultiSerializer
from kinfer.serialize.schema import get_dummy_inputs

KINFER_METADATA_KEY = "kinfer_metadata"


def add_metadata_to_onnx(
    model_proto: onnx.ModelProto,
    input_schema: P.InputSchema,
    output_schema: P.OutputSchema,
) -> onnx.ModelProto:
    """Add metadata to ONNX model.

    Args:
        model_proto: ONNX model prototype
        input_schema: Input schema to use for model export.
        output_schema: Output schema to use for model export.

    Returns:
        ONNX model with added metadata
    """
    model_schema = P.ModelSchema(input_schema=input_schema, output_schema=output_schema)
    schema_bytes = model_schema.SerializeToString()
    meta = model_proto.metadata_props.add()
    meta.key = KINFER_METADATA_KEY
    meta.value = schema_bytes
    return model_proto


def export_model(
    model: torch.jit.ScriptModule,
    input_schema: P.InputSchema,
    output_schema: P.OutputSchema,
) -> onnx.ModelProto:
    """Export PyTorch model to ONNX format with metadata.

    Args:
        model: PyTorch model to export.
        input_schema: Input schema to use for model export.
        output_schema: Output schema to use for model export.

    Returns:
        ONNX inference session
    """
    input_serializer = PyTorchMultiSerializer(input_schema)
    output_serializer = PyTorchMultiSerializer(output_schema)

    input_dummy_values = get_dummy_inputs(input_schema)
    input_tensors = input_serializer.serialize_input(input_dummy_values)

    # Attempts to run the model with the dummy inputs.
    try:
        pred_output_tensors = model(**input_tensors)
    except Exception as e:
        signature = inspect.signature(model.forward)
        model_input_names = [
            p.name for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        raise ValueError(
            f"Failed to run model with dummy inputs; input names are {model_input_names} while "
            f"input schema is {input_schema}"
        ) from e

    # Attempts to parse the output tensors using the output schema.
    if isinstance(pred_output_tensors, Tensor):
        pred_output_tensors = (pred_output_tensors,)
    if isinstance(pred_output_tensors, Sequence):
        pred_output_tensors = output_serializer.assign_output_names(pred_output_tensors)
    if not isinstance(pred_output_tensors, dict):
        raise ValueError("Output tensors could not be converted to dictionary")
    try:
        pred_output_tensors = output_serializer.deserialize_output(pred_output_tensors)
    except Exception as e:
        raise ValueError("Failed to parse output tensors using output schema; are you sure it is correct?") from e

    # Export model to buffer
    buffer = BytesIO()
    torch.onnx.export(model, input_tensors, buffer)  # type: ignore[arg-type]
    buffer.seek(0)

    # Loads the model from the buffer and adds metadata.
    model_proto = onnx.load_model(buffer)
    model_proto = add_metadata_to_onnx(model_proto, input_schema, output_schema)

    return model_proto


def get_model(model_proto: onnx.ModelProto) -> ort.InferenceSession:
    """Converts a model proto to an inference session.

    Args:
        model_proto: ONNX model proto to convert to inference session.

    Returns:
        ONNX inference session
    """
    buffer = BytesIO()
    onnx.save_model(model_proto, buffer)
    buffer.seek(0)
    return ort.InferenceSession(buffer.read())
