"""Tests serialization and deserialization to PyTorch tensors."""

import random

import pytest
import torch
from torch import Tensor

from kinfer.protos.kinfer_pb2 import AudioFrameValue, JointPosition, JointPositionsValue, TensorValue, TimestampValue
from kinfer.serialize.pytorch import PyTorchSerializer


def test_serialize_timestamp() -> None:
    serializer = PyTorchSerializer()

    # From timestamp value to tensor.
    timestamp = TimestampValue(seconds=1, nanos=500_000_000)
    tensor = serializer.serialize_timestamp(timestamp)
    assert isinstance(tensor, Tensor)
    assert tensor.item() == 1.5

    # Back to timestamp value.
    new_timestamp = serializer.deserialize_timestamp(tensor)
    assert isinstance(new_timestamp, TimestampValue)
    assert new_timestamp.seconds == timestamp.seconds
    assert new_timestamp.nanos == timestamp.nanos


def test_serialize_tensor() -> None:
    serializer = PyTorchSerializer()

    # From tensor to tensor value.
    tensor = torch.randn(1, 2, 3)
    tensor_value = serializer.deserialize_tensor(tensor)
    assert isinstance(tensor_value, TensorValue)
    assert tensor_value.shape == [1, 2, 3]

    # From tensor value to tensor.
    new_tensor = serializer.serialize_tensor(tensor_value)
    assert isinstance(new_tensor, Tensor)
    assert torch.allclose(tensor, new_tensor)


@pytest.mark.parametrize("radians", [True, False])
def test_serialize_joint_positions(radians: bool) -> None:
    serializer = PyTorchSerializer()

    joints = {"joint_3": 0.5, "joint_1": 1.0, "joint_2": 1.5}

    # From joint positions to tensor.
    joint_positions = JointPositionsValue()
    for name, degrees in joints.items():
        joint_positions.positions.append(JointPosition(joint_name=name, degrees=degrees))
    tensor, names = serializer.serialize_joint_positions(joint_positions, radians=radians)
    assert isinstance(tensor, Tensor)

    # Back to joint positions value.
    new_joint_positions = serializer.deserialize_joint_positions(names, tensor, radians=radians)
    assert isinstance(new_joint_positions, JointPositionsValue)
    for joint in new_joint_positions.positions:
        assert joint.joint_name in joints
        assert joint.degrees == joints[joint.joint_name]


@pytest.mark.parametrize("target_sample_rate", [16000, None])
def test_serialize_audio_frame(target_sample_rate: int | None) -> None:
    serializer = PyTorchSerializer()

    audio_frame = AudioFrameValue(channels=2, sample_rate=44100, data=[random.random() for _ in range(20000)])
    tensor = serializer.serialize_audio_frame(audio_frame, sample_rate=target_sample_rate)
    assert isinstance(tensor, Tensor)

    # Back to audio frame value.
    new_audio_frame = serializer.deserialize_audio_frame(tensor, sample_rate=audio_frame.sample_rate)
    assert isinstance(new_audio_frame, AudioFrameValue)
    assert new_audio_frame.channels == audio_frame.channels

    if target_sample_rate is None:
        assert new_audio_frame.data == audio_frame.data
