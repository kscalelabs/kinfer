"""PyTorch model export utilities."""

__all__ = [
    "export_fn",
]

import io
from typing import cast

import onnx
import torch
from onnx.onnx_pb import ModelProto
from torch._C import FunctionSchema

from kinfer.rust_bindings import PyInputType, PyModelMetadata


def export_fn(
    model: torch.jit.ScriptFunction,
    metadata: PyModelMetadata,
) -> ModelProto:
    """Exports a PyTorch function to ONNX.

    Args:
        model: The model to export.
        metadata: The metadata for the model.

    Returns:
        The ONNX model as a `ModelProto`.
    """
    if not isinstance(model, torch.jit.ScriptFunction):
        raise ValueError("Model must be a torch.jit.ScriptFunction")

    schema = cast(FunctionSchema, model.schema)
    input_names = [arg.name for arg in schema.arguments]

    # Gets the dummy input tensors for exporting the model.
    args = []
    for name in input_names:
        shape = PyInputType(name).get_shape(metadata)
        args.append(torch.zeros(shape))

    buffer = io.BytesIO()
    torch.onnx.export(
        model=model,
        f=buffer,  # type: ignore[arg-type]
        args=tuple(args),
        input_names=input_names,
        external_data=False,
    )
    buffer.seek(0)
    model_bytes = buffer.read()
    return onnx.load_from_string(model_bytes)
