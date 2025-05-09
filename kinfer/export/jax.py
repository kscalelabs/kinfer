"""Jax model export utilities."""

import inspect
import io
import logging

import tensorflow as tf
import tf2onnx
from equinox.internal._finalise_jaxpr import finalise_fn
from jax.experimental import jax2tf
from jaxlib.xla_extension import PjitFunction

from kinfer.export.common import get_shape

logger = logging.getLogger(__name__)


def export_fn(
    model: PjitFunction,
    *,
    num_joints: int | None = None,
    carry_shape: tuple[int, ...] | None = None,
    opset: int = 13,
) -> bytes:
    """Export a JAX function to ONNX."""
    if not isinstance(model, PjitFunction):
        raise ValueError("Model must be a PjitFunction")

    params = inspect.signature(model).parameters
    input_names = list(params.keys())

    # Gets the dummy input tensors for exporting the model.
    tf_args = []
    for name in input_names:
        shape = get_shape(
            name,
            num_joints=num_joints,
            carry_shape=carry_shape,
        )
        tf_args.append(tf.TensorSpec(shape, tf.float32))

    finalised_fn = finalise_fn(model)
    tf_fn = tf.function(jax2tf.convert(finalised_fn, enable_xla=False))

    model_proto, external_tensor_storage = tf2onnx.convert.from_function(
        tf_fn,
        input_signature=tf_args,
        opset=opset,
        large_model=True,
    )
    buffer = io.BytesIO()
    tf2onnx.utils.save_onnx_zip(buffer, model_proto, external_tensor_storage)
    buffer.seek(0)
    return buffer.read()
