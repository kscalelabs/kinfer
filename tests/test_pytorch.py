"""Tests for model inference functionality on a PyTorch model."""

import logging
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime
import torch
from torch import Tensor

from kinfer.common.types import Metadata
from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import ModelProviderABC, PyModelRunner, PyModelRuntime

logger = logging.getLogger(__name__)

JOINT_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]
NUM_JOINTS = len(JOINT_NAMES)
CARRY_SIZE = 10
NUM_COMMANDS = 4


@torch.jit.script
def init_fn() -> Tensor:
    return torch.zeros((10,))


@torch.jit.script
def step_fn(
    joint_angles: Tensor,
    joint_angular_velocities: Tensor,
    projected_gravity: Tensor,
    accelerometer: Tensor,
    gyroscope: Tensor,
    command: Tensor,
    carry: Tensor,
) -> tuple[Tensor, Tensor]:
    output = (
        joint_angles.mean()
        + joint_angular_velocities.mean()
        + projected_gravity.mean()
        + accelerometer.mean()
        + gyroscope.mean()
        + command.mean()
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
        num_commands=NUM_COMMANDS,
        carry_shape=(CARRY_SIZE,),
    )

    kinfer_model = pack(
        init_fn_onnx,
        step_fn_onnx,
        joint_names=JOINT_NAMES,
        num_commands=NUM_COMMANDS,
        carry_shape=(CARRY_SIZE,),
    )

    # Saves the model to disk.
    root_dir = Path(tmpdir)
    (kinfer_path := root_dir / "model.kinfer").write_bytes(kinfer_model)

    # Ensures that we can open the file like a regular tar file.
    with tarfile.open(kinfer_path, "r:gz") as tar:
        assert tar.getnames() == ["init_fn.onnx", "step_fn.onnx", "metadata.json"]

        # Checks that joint_names.json is valid JSON.
        if (fpath := tar.extractfile("metadata.json")) is None:
            raise ValueError("metadata.json not found")
        metadata = Metadata.model_validate_json(fpath.read().decode("utf-8"))
        assert metadata.joint_names == JOINT_NAMES

        # Validates that we can construct a session in Python.
        if (fpath := tar.extractfile("init_fn.onnx")) is None:
            raise ValueError("init_fn.onnx not found")
        init_session = onnxruntime.InferenceSession(fpath.read())
        assert init_session.get_modelmeta().graph_name == "main_graph"
        if (fpath := tar.extractfile("step_fn.onnx")) is None:
            raise ValueError("step_fn.onnx not found")
        step_session = onnxruntime.InferenceSession(fpath.read())
        assert step_session.get_modelmeta().graph_name == "main_graph"

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

        def get_command(self) -> np.ndarray:
            return np.random.randn(NUM_COMMANDS)

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

    # Tests the runtime, which runs in a separate Rust thread.
    dt = 10
    model_runtime = PyModelRuntime(model_runner, dt)
    model_runtime.start()
    time.sleep(dt * 4.5 / 1000)
    model_runtime.stop()
    assert num_actions == 8, f"num_actions: {num_actions}"


if __name__ == "__main__":
    # python -m tests.test_pytorch
    with tempfile.TemporaryDirectory() as tmpdir:
        test_export(Path(tmpdir))
