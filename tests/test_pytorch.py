"""Tests for model inference functionality on a PyTorch model."""

import logging

import numpy as np
import torch
from torch import Tensor

from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import ModelProviderABC, PyModelRunner

logger = logging.getLogger(__name__)

JOINT_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]
NUM_JOINTS = len(JOINT_NAMES)
CARRY_SIZE = 10


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
    carry: Tensor,
) -> tuple[Tensor, Tensor]:
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


class DummyModelProvider(ModelProviderABC):
    def get_joint_angles(self, joint_names: list[str]) -> np.ndarray[np.float32]:
        assert len(joint_names) == NUM_JOINTS
        return np.random.randn(NUM_JOINTS)

    def get_joint_angular_velocities(self, joint_names: list[str]) -> np.ndarray[np.float32]:
        assert len(joint_names) == NUM_JOINTS
        return np.random.randn(NUM_JOINTS)

    def get_projected_gravity(self) -> np.ndarray[np.float32]:
        return np.random.randn(3)

    def get_accelerometer(self) -> np.ndarray[np.float32]:
        return np.random.randn(3)

    def get_gyroscope(self) -> np.ndarray[np.float32]:
        return np.random.randn(3)

    def take_action(self, action: np.ndarray[np.float32]) -> None:
        logger.info("Taking action: %s", action)


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

    model_provider = DummyModelProvider()

    model_runner = PyModelRunner(kinfer_model, model_provider)

    carry = model_runner.init()
    assert carry.shape == (10,)
    for _ in range(3):
        output, carry = model_runner.step(carry)
        assert output.shape == (10,)
        assert carry.shape == (10,)


if __name__ == "__main__":
    # python -m tests.test_pytorch
    test_export()
