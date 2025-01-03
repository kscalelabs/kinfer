"""PyTorch model export utilities."""

import base64
import inspect
from io import BytesIO
from typing import Sequence

import onnx
import onnxruntime as ort
import torch
from torch import Tensor

from kinfer import proto as K
from kinfer.serialize.pytorch import PyTorchMultiSerializer
from kinfer.serialize.schema import get_dummy_io
from kinfer.serialize.utils import check_names_match

KINFER_METADATA_KEY = "kinfer_metadata"


def _add_metadata_to_onnx(model_proto: onnx.ModelProto, schema: K.ModelSchema) -> onnx.ModelProto:
    """Add metadata to ONNX model.

    Args:
        model_proto: ONNX model prototype
        schema: Model schema to use for model export.

    Returns:
        ONNX model with added metadata
    """
    schema_bytes = schema.SerializeToString()

    meta = model_proto.metadata_props.add()
    meta.key = KINFER_METADATA_KEY
    meta.value = base64.b64encode(schema_bytes).decode("utf-8")

    return model_proto


def export_model(model: torch.jit.ScriptModule, schema: K.ModelSchema) -> onnx.ModelProto:
    """Export PyTorch model to ONNX format with metadata.

    Args:
        model: PyTorch model to export.
        schema: Model schema to use for model export.

    Returns:
        ONNX inference session
    """
    # Matches each input name to the input values.
    signature = inspect.signature(model.forward)
    model_input_names = [
        p.name for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    if len(model_input_names) != len(schema.input_schema.values):
        raise ValueError(f"Expected {len(model_input_names)} inputs, but schema has {len(schema.input_schema.values)}")
    input_schema_names = [i.value_name for i in schema.input_schema.values]
    output_schema_names = [o.value_name for o in schema.output_schema.values]

    if model_input_names != input_schema_names:
        raise ValueError(f"Expected input names {model_input_names} to match schema names {input_schema_names}")

    input_serializer = PyTorchMultiSerializer(schema.input_schema)
    output_serializer = PyTorchMultiSerializer(schema.output_schema)

    input_dummy_values = get_dummy_io(schema.input_schema)
    input_tensors = input_serializer.serialize_io(input_dummy_values, as_dict=True)

    check_names_match("model_input_names", model_input_names, "input_schema", list(input_tensors.keys()))
    input_tensor_list = [input_tensors[name] for name in model_input_names]

    # Attempts to run the model with the dummy inputs.
    try:
        pred_output_tensors = model(*input_tensor_list)
    except Exception as e:
        signature = inspect.signature(model.forward)
        model_input_names = [
            p.name for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]

        raise ValueError(
            f"Failed to run model with dummy inputs; input names are {model_input_names} while "
            f"input schema is {schema.input_schema}"
        ) from e

    # Attempts to parse the output tensors using the output schema.
    if isinstance(pred_output_tensors, Tensor):
        pred_output_tensors = (pred_output_tensors,)
    if isinstance(pred_output_tensors, Sequence):
        pred_output_tensors = output_serializer.assign_names(pred_output_tensors)
    if not isinstance(pred_output_tensors, dict):
        raise ValueError("Output tensors could not be converted to dictionary")
    try:
        pred_output_tensors = output_serializer.deserialize_io(pred_output_tensors)
    except Exception as e:
        raise ValueError("Failed to parse output tensors using output schema; are you sure it is correct?") from e

    # Export model to buffer
    buffer = BytesIO()
    torch.onnx.export(
        model=model,
        f=buffer,  # type: ignore[arg-type]
        kwargs=input_tensors,
        input_names=input_schema_names,
        output_names=output_schema_names,
    )
    buffer.seek(0)

    # Loads the model from the buffer and adds metadata.
    model_proto = onnx.load_model(buffer)
    model_proto = _add_metadata_to_onnx(model_proto, schema)

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
