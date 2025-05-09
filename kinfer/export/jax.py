"""Jax model export utilities."""

import logging
from pathlib import Path
from typing import Callable

import tensorflow as tf
import tf2onnx
from equinox.internal._finalise_jaxpr import finalise_fn
from jax.experimental import jax2tf

logger = logging.getLogger(__name__)


def export_onnx(
    model: Callable,
    input_shapes: list[tuple[int, ...]],
    output_dir: str | Path = "export",
    opset: int = 13,
) -> None:
    """Export a JAX function to ONNX."""
    finalised_fn = finalise_fn(model)
    tf_fn = tf.function(jax2tf.convert(finalised_fn, enable_xla=False))
    tf_args = [tf.TensorSpec(input_shape, tf.float32) for input_shape in input_shapes]

    logger.info("Exporting model to %s", output_dir)
    _ = tf2onnx.convert.from_function(
        tf_fn,
        input_signature=tf_args,
        opset=opset,
        output_path=output_dir,
    )
