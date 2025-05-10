"""Tests for model inference functionality on a PyTorch model."""

import torch
from torch import Tensor

from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack


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


if __name__ == "__main__":
    # python -m tests.test_pytorch
    test_export()
