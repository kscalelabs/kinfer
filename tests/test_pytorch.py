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

from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import ModelProviderABC, PyModelMetadata, PyModelRunner, PyModelRuntime, metadata_from_json

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
    time: Tensor,
    carry: Tensor,
) -> tuple[Tensor, Tensor]:
    output = (
        joint_angles.mean()
        + joint_angular_velocities.mean()
        + projected_gravity.mean()
        + accelerometer.mean()
        + gyroscope.mean()
        + command.mean()
        + torch.cos(time).mean()
        + torch.sin(time).mean()
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

    # Ensures that we can open the file like a regular tar file.
    with tarfile.open(kinfer_path, "r:gz") as tar:
        assert tar.getnames() == ["init_fn.onnx", "step_fn.onnx", "metadata.json"]

        # Checks that joint_names.json is valid JSON.
        if (fpath := tar.extractfile("metadata.json")) is None:
            raise ValueError("metadata.json not found")
        metadata = metadata_from_json(fpath.read().decode("utf-8"))
        assert metadata.joint_names == JOINT_NAMES  # type: ignore[attr-defined]

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
