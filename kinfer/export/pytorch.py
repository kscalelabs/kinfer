"""PyTorch model export utilities."""

__all__ = [
    "export_fn",
]

import io
from typing import cast

import torch
from torch._C import FunctionSchema

from kinfer.export.common import get_shape


def export_fn(
    model: torch.jit.ScriptFunction,
    *,
    num_joints: int | None = None,
    carry_shape: tuple[int, ...] | None = None,
) -> bytes:
    """Exports a PyTorch function to ONNX.

    Args:
        model: The model to export.
        num_joints: The number of joints in the model.
        carry_shape: The shape of the carry tensor.

    Returns:
        A BytesIO object containing the ONNX model.
    """
    if not isinstance(model, torch.jit.ScriptFunction):
        raise ValueError("Model must be a torch.jit.ScriptFunction")

    schema = cast(FunctionSchema, model.schema)
    input_names = [arg.name for arg in schema.arguments]

    # Gets the dummy input tensors for exporting the model.
    args = []
    for name in input_names:
        shape = get_shape(
            name,
            num_joints=num_joints,
            carry_shape=carry_shape,
        )
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
    return buffer.read()
