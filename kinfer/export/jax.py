"""Jax model export utilities."""

import inspect
import logging

import tensorflow as tf
import tf2onnx
from equinox.internal._finalise_jaxpr import finalise_fn
from jax._src.stages import Wrapped
from jax.experimental import jax2tf
from onnx.onnx_pb import ModelProto

from kinfer.rust_bindings import PyInputType, PyModelMetadata

logger = logging.getLogger(__name__)


def export_fn(
    model: Wrapped,
    metadata: PyModelMetadata,
    *,
    opset: int = 13,
) -> ModelProto:
    """Export a JAX function to ONNX.

    Args:
        model: The model to export.
        metadata: The metadata for the model.
        opset: The ONNX opset to use.

    Returns:
        The ONNX model as a `ModelProto`.
    """
    if not isinstance(model, Wrapped):
        raise ValueError("Model must be a Wrapped function")

    params = inspect.signature(model).parameters
    input_names = list(params.keys())

    # Gets the dummy input tensors for exporting the model.
    tf_args = []
    for name in input_names:
        shape = PyInputType(name).get_shape(metadata)
        tf_args.append(tf.TensorSpec(shape, tf.float32, name=name))

    finalised_fn = finalise_fn(model)
    tf_fn = tf.function(jax2tf.convert(finalised_fn, enable_xla=False))

    model_proto, _ = tf2onnx.convert.from_function(
        tf_fn,
        input_signature=tf_args,
        opset=opset,
        large_model=False,
    )
    return model_proto
