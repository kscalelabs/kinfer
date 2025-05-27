"""Functions for serializing and deserializing models."""

__all__ = [
    "pack",
]

import io
import logging
import tarfile

from onnx.onnx_pb import ModelProto

from kinfer.rust_bindings import PyInputType, PyModelMetadata

logger = logging.getLogger(__name__)


def pack(
    init_fn: ModelProto,
    step_fn: ModelProto,
    metadata: PyModelMetadata,
) -> bytes:
    """Packs the initialization function and step function into a directory.

    Args:
        init_fn: The initialization function.
        step_fn: The step function.
        metadata: The metadata for the model.
    """
    num_joints = len(metadata.joint_names)  # type: ignore[attr-defined]

    # Checks the `init` function.
    if len(init_fn.graph.input) > 0:
        raise ValueError(f"`init` function should not have any inputs! Got {len(init_fn.graph.input)}")
    if len(init_fn.graph.output) != 1:
        raise ValueError(f"`init` function should have exactly 1 output! Got {len(init_fn.graph.output)}")
    init_carry = init_fn.graph.output[0]
    init_carry_shape = tuple(dim.dim_value for dim in init_carry.type.tensor_type.shape.dim)

    if metadata.carry_size != init_carry_shape:  # type: ignore[attr-defined]
        logger.warning(
            "Updating carry size from %s to %s to match the `init` function",
            metadata.carry_size,  # type: ignore[attr-defined]
            init_carry_shape,
        )
        metadata.carry_size = init_carry_shape  # type: ignore[attr-defined]

    # Checks the `step` function.
    for step_input in step_fn.graph.input:
        step_input_type = step_input.type.tensor_type
        shape = tuple(dim.dim_value for dim in step_input_type.shape.dim)
        expected_shape = PyInputType(step_input.name).get_shape(metadata)
        if shape != tuple(expected_shape):
            raise ValueError(f"Expected shape {expected_shape} for input `{step_input.name}`, got {shape}")

    if len(step_fn.graph.output) != 2:
        raise ValueError(f"Step function must have exactly 2 outputs, got {len(step_fn.graph.output)}")

    output_actions = step_fn.graph.output[0]
    actions_shape = tuple(dim.dim_value for dim in output_actions.type.tensor_type.shape.dim)
    if actions_shape != (num_joints,):
        raise ValueError(f"Expected output shape {num_joints} for output `{output_actions.name}`, got {actions_shape}")

    output_carry = step_fn.graph.output[1]
    output_carry_shape = tuple(dim.dim_value for dim in output_carry.type.tensor_type.shape.dim)
    if output_carry_shape != init_carry_shape:
        raise ValueError(f"Expected carry shape {init_carry_shape} for output carry, got {output_carry_shape}")

    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add_file_bytes(name: str, data: bytes) -> None:  # noqa: ANN401
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add_file_bytes("init_fn.onnx", init_fn.SerializeToString())
        add_file_bytes("step_fn.onnx", step_fn.SerializeToString())
        add_file_bytes("metadata.json", metadata.to_json().encode("utf-8"))

    buffer.seek(0)

    return buffer.read()
