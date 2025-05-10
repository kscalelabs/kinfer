"""Tests for model inference functionality on a JAX model."""

import jax
from jax import numpy as jnp

from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import get_version


@jax.jit
def init_fn() -> jnp.ndarray:
    return jnp.zeros((10,))


@jax.jit
def step_fn(
    joint_angles: jnp.ndarray,
    joint_angular_velocities: jnp.ndarray,
    projected_gravity: jnp.ndarray,
    accelerometer: jnp.ndarray,
    gyroscope: jnp.ndarray,
    carry: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    output = (
        joint_angles.mean()
        + joint_angular_velocities.mean()
        + projected_gravity.mean()
        + accelerometer.mean()
        + gyroscope.mean()
        + carry.mean()
    ) * joint_angles
    next_carry = carry + 1
    return output, next_carry


def test_export() -> None:
    joint_names = ["left_arm", "right_arm", "left_leg", "right_leg"]

    init_fn_onnx = export_fn(
        model=init_fn,
    )

    step_fn_onnx = export_fn(
        model=step_fn,
        num_joints=len(joint_names),
        carry_shape=(10,),
    )

    kinfer_model = pack(
        init_fn_onnx,
        step_fn_onnx,
        joint_names=joint_names,
        carry_shape=(10,),
    )

    print("Version:", get_version())


if __name__ == "__main__":
    # python -m tests.test_jax
    test_export()
