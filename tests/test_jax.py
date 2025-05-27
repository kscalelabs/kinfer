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
from kinfer.rust_bindings import ModelProviderABC, PyModelMetadata, PyModelRunner

logger = logging.getLogger(__name__)

JOINT_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]
NUM_JOINTS = len(JOINT_NAMES)
CARRY_SIZE = 10
NUM_COMMANDS = 4


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
    command: jnp.ndarray,
    time: jnp.ndarray,
    carry: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    output = (
        joint_angles.mean()
        + joint_angular_velocities.mean()
        + projected_gravity.mean()
        + accelerometer.mean()
        + gyroscope.mean()
        + command.mean()
        + jnp.cos(time).mean()
        + jnp.sin(time).mean()
        + carry.mean()
    ) * joint_angles
    next_carry = carry + 1
    return output, next_carry


def test_export(tmpdir: Path) -> None:
    metadata = PyModelMetadata(
        joint_names=JOINT_NAMES,
        num_commands=NUM_COMMANDS,
        carry_size=[CARRY_SIZE],
    )

    init_fn_onnx = export_fn(init_fn, metadata)
    step_fn_onnx = export_fn(step_fn, metadata)
    kinfer_model = pack(init_fn_onnx, step_fn_onnx, metadata)

    # Saves the model to disk.
    root_dir = Path(tmpdir)
    (kinfer_path := root_dir / "model.kinfer").write_bytes(kinfer_model)

    num_actions = 0

    class DummyModelProvider(ModelProviderABC):
        def get_inputs(self, input_types: Sequence[str], metadata: PyModelMetadata) -> dict[str, np.ndarray]:
            return_values: dict[str, np.ndarray] = {}
            for input_type in input_types:
                match input_type:
                    case "joint_angles":
                        return_values["joint_angles"] = np.random.randn(NUM_JOINTS)
                    case "joint_angular_velocities":
                        return_values["joint_angular_velocities"] = np.random.randn(NUM_JOINTS)
                    case "projected_gravity":
                        return_values["projected_gravity"] = np.random.randn(3)
                    case "accelerometer":
                        return_values["accelerometer"] = np.random.randn(3)
                    case "gyroscope":
                        return_values["gyroscope"] = np.random.randn(3)
                    case "command":
                        return_values["command"] = np.random.randn(NUM_COMMANDS)
                    case "time":
                        return_values["time"] = np.random.randn(1)
                    case _:
                        raise ValueError(f"Unknown input type: {input_type}")
            return return_values

        def take_action(self, action: np.ndarray, metadata: PyModelMetadata) -> None:
            assert metadata.joint_names == JOINT_NAMES  # type: ignore[attr-defined]
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
