"""Tests for model inference functionality on a JAX model."""

import logging
import tempfile
from pathlib import Path
from typing import Sequence

import jax
import numpy as np
from jax import numpy as jnp

from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import ModelProviderABC, PyModelRunner

logger = logging.getLogger(__name__)

JOINT_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]
NUM_JOINTS = len(JOINT_NAMES)
CARRY_SIZE = 10


@jax.jit
def init_fn() -> jnp.ndarray:
    return jnp.zeros((CARRY_SIZE,))


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


def test_export(tmpdir: Path) -> None:
    init_fn_onnx = export_fn(
        model=init_fn,
    )

    step_fn_onnx = export_fn(
        model=step_fn,
        num_joints=NUM_JOINTS,
        carry_shape=(CARRY_SIZE,),
    )

    kinfer_model = pack(
        init_fn_onnx,
        step_fn_onnx,
        joint_names=JOINT_NAMES,
        carry_shape=(CARRY_SIZE,),
    )

    # Saves the model to disk.
    root_dir = Path(tmpdir)
    (kinfer_path := root_dir / "model.kinfer").write_bytes(kinfer_model)

    num_actions = 0

    class DummyModelProvider(ModelProviderABC):
        def get_joint_angles(self, joint_names: Sequence[str]) -> np.ndarray:
            assert len(joint_names) == NUM_JOINTS
            return np.random.randn(NUM_JOINTS)

        def get_joint_angular_velocities(self, joint_names: Sequence[str]) -> np.ndarray:
            assert len(joint_names) == NUM_JOINTS
            return np.random.randn(NUM_JOINTS)

        def get_projected_gravity(self) -> np.ndarray:
            return np.random.randn(3)

        def get_accelerometer(self) -> np.ndarray:
            return np.random.randn(3)

        def get_gyroscope(self) -> np.ndarray:
            return np.random.randn(3)

        def take_action(self, joint_names: Sequence[str], action: np.ndarray) -> None:
            assert joint_names == JOINT_NAMES
            assert action.shape == (NUM_JOINTS,)
            nonlocal num_actions
            num_actions += 1

    # Creates a model runner from the kinfer model.
    model_provider = DummyModelProvider()
    model_runner = PyModelRunner(str(kinfer_path), model_provider)

    carry = model_runner.init()
    assert carry.shape == (CARRY_SIZE,)
    for _ in range(3):
        output, carry = model_runner.step(carry)
        model_runner.take_action(output)
        assert carry.shape == (CARRY_SIZE,), f"Carry shape: {carry.shape}"

    assert num_actions == 3


if __name__ == "__main__":
    # python -m tests.test_jax
    with tempfile.TemporaryDirectory() as tmpdir:
        test_export(Path(tmpdir))
