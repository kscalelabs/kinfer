"""Defines a serializer for PyTorch tensors."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from kinfer.protos.kinfer_pb2 import (
    AudioFrameValue,
    CameraFrameValue,
    JointPosition,
    JointPositionsValue,
    JointTorquesValue,
    TensorValue,
    TimestampValue,
)
from kinfer.serialize.base import (
    AudioFrameSerializer,
    CameraFrameSerializer,
    JointPositionsSerializer,
    JointTorquesSerializer,
    Serializer,
    TensorSerializer,
    TimestampSerializer,
)


class PyTorchBaseSerializer:
    def __init__(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype


class PyTorchTimestampSerializer(PyTorchBaseSerializer, TimestampSerializer[Tensor]):
    def serialize_timestamp(self, timestamp: TimestampValue) -> Tensor:
        elapsed_seconds = timestamp.seconds - timestamp.start_seconds
        elapsed_nanos = timestamp.nanos - timestamp.start_nanos
        if elapsed_nanos < 0:
            elapsed_seconds -= 1
            elapsed_nanos += 1_000_000_000
        total_elapsed_seconds = elapsed_seconds + elapsed_nanos / 1_000_000_000
        return torch.tensor([total_elapsed_seconds], dtype=self.dtype, device=self.device, requires_grad=False)

    def deserialize_timestamp(self, value: Tensor) -> TimestampValue:
        total_elapsed_seconds = value.item()
        elapsed_seconds = int(total_elapsed_seconds)
        elapsed_nanos = int((total_elapsed_seconds - elapsed_seconds) * 1_000_000_000)
        return TimestampValue(seconds=elapsed_seconds, nanos=elapsed_nanos)


class PyTorchTensorSerializer(PyTorchBaseSerializer, TensorSerializer[Tensor]):
    def serialize_tensor(self, value: TensorValue) -> Tensor:
        tensor = torch.tensor(value.data, dtype=self.dtype, device=self.device)
        tensor = tensor.view(*value.shape)
        return tensor

    def deserialize_tensor(self, tensor: Tensor) -> TensorValue:
        tensor_data = tensor.detach().cpu().flatten().numpy().tolist()
        tensor_shape = list(tensor.size())
        tensor_value = TensorValue()
        tensor_value.data.extend(tensor_data)
        tensor_value.shape.extend(tensor_shape)
        return tensor_value


class PyTorchJointPositionsSerializer(PyTorchBaseSerializer, JointPositionsSerializer[Tensor]):
    def serialize_joint_positions(
        self,
        joint_positions: JointPositionsValue,
        *,
        radians: bool = False,
    ) -> tuple[Tensor, list[str]]:
        names, values = zip(
            *[
                (joint.joint_name, joint.degrees)
                for joint in sorted(joint_positions.positions, key=lambda x: x.joint_name)
            ]
        )
        tensor = torch.tensor(values, dtype=self.dtype, device=self.device)
        if radians:
            tensor *= math.pi / 180
        return tensor, list(names)

    def deserialize_joint_positions(
        self,
        names: list[str],
        value: Tensor,
        *,
        radians: bool = False,
    ) -> JointPositionsValue:
        if radians:
            value *= 180 / math.pi
        joint_positions = JointPositionsValue()
        for name, degrees in zip(names, value.tolist()):
            joint_positions.positions.append(JointPosition(joint_name=name, degrees=degrees))
        return joint_positions


class PyTorchAudioFrameSerializer(PyTorchBaseSerializer, AudioFrameSerializer[Tensor]):
    def serialize_audio_frame(self, audio_frame: AudioFrameValue, *, sample_rate: int | None = None) -> Tensor:
        tensor = torch.tensor(audio_frame.data, dtype=self.dtype, device=self.device)
        tensor = tensor.view(audio_frame.channels, -1)
        if sample_rate is not None and sample_rate != audio_frame.sample_rate:
            tensor = F.interpolate(tensor[None, :, :], scale_factor=sample_rate / audio_frame.sample_rate)[0]
        return tensor

    def deserialize_audio_frame(self, value: Tensor, sample_rate: int) -> AudioFrameValue:
        audio_data = value.cpu().flatten().numpy().tolist()
        audio_frame = AudioFrameValue()
        audio_frame.channels = value.size(0)
        audio_frame.sample_rate = sample_rate
        audio_frame.data.extend(audio_data)
        return audio_frame


class PyTorchCameraFrameSerializer(PyTorchBaseSerializer, CameraFrameSerializer[Tensor]):
    def serialize_camera_frame(self, camera_frame: CameraFrameValue) -> Tensor:
        # Implement serialization logic for CameraFrame
        pass

    def deserialize_camera_frame(self, value: Tensor) -> CameraFrameValue:
        # Implement deserialization logic for CameraFrameValue
        pass


class PyTorchJointTorquesSerializer(PyTorchBaseSerializer, JointTorquesSerializer[Tensor]):
    def serialize_joint_torques(self, joint_torques: JointTorquesValue) -> Tensor:
        # Implement serialization logic for JointTorques
        pass

    def deserialize_joint_torques(self, value: Tensor) -> JointTorquesValue:
        # Implement deserialization logic for JointTorquesValue
        pass


class PyTorchSerializer(
    PyTorchTimestampSerializer,
    PyTorchTensorSerializer,
    PyTorchJointPositionsSerializer,
    PyTorchAudioFrameSerializer,
    PyTorchCameraFrameSerializer,
    PyTorchJointTorquesSerializer,
    Serializer[Tensor],
):
    pass
