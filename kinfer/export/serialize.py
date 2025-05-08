"""Functions for serializing and deserializing models."""

__all__ = [
    "pack",
]

import io
import json
import tarfile

import onnx

from kinfer.export.common import get_shape


def pack(
    init_fn: bytes,
    step_fn: bytes,
    joint_names: list[str],
    carry_shape: tuple[int, ...],
) -> bytes:
    """Packs the initialization function and step function into a tarfile.

    Args:
        init_fn: The initialization function.
        step_fn: The step function.
        joint_names: The list of joint names, in the order that the model
            expects them to be provided.
        carry_shape: The shape of the carry tensor.

    Returns:
        A BytesIO object containing the tarfile.
    """
    num_joints = len(joint_names)

    # Load and validate ONNX models
    init_model = onnx.load_from_string(init_fn)
    step_model = onnx.load_from_string(step_fn)

    # Checks the `init` function.
    if len(init_model.graph.input) > 0:
        raise ValueError(f"`init` function should not have any inputs! Got {len(init_model.graph.input)}")
    if len(init_model.graph.output) != 1:
        raise ValueError(f"`init` function should have exactly 1 output! Got {len(init_model.graph.output)}")

    # Checks the `step` function.
    for step_input in step_model.graph.input:
        step_input_type = step_input.type.tensor_type
        shape = tuple(dim.dim_value for dim in step_input_type.shape.dim)
        expected_shape = get_shape(
            step_input.name,
            num_joints=num_joints,
            carry_shape=carry_shape,
        )
        if shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape} for input `{step_input.name}`, got {shape}")

    if len(step_model.graph.output) != 2:
        raise ValueError(f"Step function must have exactly 2 outputs, got {len(step_model.graph.output)}")

    model_output = step_model.graph.output[0]
    output_shape = tuple(dim.dim_value for dim in model_output.type.tensor_type.shape.dim)
    if output_shape != (num_joints,):
        raise ValueError(f"Expected output shape {num_joints} for output `{model_output.name}`, got {output_shape}")

    model_carry = step_model.graph.output[1]
    output_carry_shape = tuple(dim.dim_value for dim in model_carry.type.tensor_type.shape.dim)
    if output_carry_shape != carry_shape:
        raise ValueError(f"Expected carry shape {carry_shape} for output `{model_carry.name}`, got {carry_shape}")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add_file_bytes(name: str, data: bytes) -> None:  # noqa: ANN401
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add_file_bytes("init_fn.onnx", init_fn)
        add_file_bytes("step_fn.onnx", step_fn)
        add_file_bytes("joint_names.json", json.dumps(joint_names).encode("utf-8"))

    buffer.seek(0)
    return buffer.read()
