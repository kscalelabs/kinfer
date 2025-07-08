"""Core transport envelope and payload schema definitions."""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from kinfer.rust_bindings import PyCommandField


class PayloadType(Enum):
    """Supported payload types for command data."""
    FLOAT_VECTOR = "float_vector"
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    PROTO = "proto"
    BINARY = "binary"
    JSON = "json"
    CUSTOM = "custom"


@dataclass
class TransportEnvelope:
    """Transport envelope - HOW to decode the data."""
    payload_type: PayloadType
    payload_length: Optional[int] = None
    codec_info: Optional[Dict[str, Any]] = None


@dataclass
class PayloadSchema:
    """Payload schema - WHAT the decoded data means."""
    fields: List[PyCommandField]
    total_length: int
    description: str
    custom_properties: Optional[Dict[str, Any]] = None


@dataclass
class CommandDefinition:
    """Complete command definition with transport and payload info."""
    command_type: str
    transport_envelope: TransportEnvelope
    payload_schema: PayloadSchema
    kinfer_version: str = "1.0"