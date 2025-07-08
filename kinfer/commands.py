"""Command infrastructure and base registry."""

from typing import Dict, List, Optional, Any
from kinfer.transport import CommandDefinition, TransportEnvelope, PayloadSchema, PayloadType
from kinfer.rust_bindings import PyCommandTypeInfo, PyCommandField, PyJointBias


class CommandTypeRegistry:
    """Base registry for command types with transport envelope and payload schema."""
    
    @classmethod
    def get_all_commands(cls) -> Dict[str, CommandDefinition]:
        """Get all registered command definitions. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_all_commands()")
    
    @classmethod
    def get_command_definition(cls, command_type: str) -> Optional[CommandDefinition]:
        """Get complete command definition."""
        return cls.get_all_commands().get(command_type)
    
    @classmethod
    def get_transport_envelope(cls, command_type: str) -> Optional[TransportEnvelope]:
        """Get transport envelope for a command type."""
        cmd_def = cls.get_command_definition(command_type)
        return cmd_def.transport_envelope if cmd_def else None
    
    @classmethod
    def get_payload_schema(cls, command_type: str) -> Optional[PayloadSchema]:
        """Get payload schema for a command type."""
        cmd_def = cls.get_command_definition(command_type)
        return cmd_def.payload_schema if cmd_def else None
    
    @classmethod
    def list_command_types(cls) -> List[str]:
        """List all registered command types."""
        return list(cls.get_all_commands().keys())
    
    @classmethod
    def validate_command_type(cls, command_type: str) -> bool:
        """Check if a command type is registered."""
        return command_type in cls.get_all_commands()
    
    @classmethod
    def get_decoder_info(cls, command_type: str) -> Optional[Dict[str, Any]]:
        """Get complete decoder information for deployment."""
        cmd_def = cls.get_command_definition(command_type)
        if not cmd_def:
            return None
        
        return {
            "command_type": cmd_def.command_type,
            "transport_envelope": {
                "payload_type": cmd_def.transport_envelope.payload_type.value,
                "payload_length": cmd_def.transport_envelope.payload_length,
                "codec_info": cmd_def.transport_envelope.codec_info,
            },
            "payload_schema": {
                "fields": [
                    {
                        "name": field.name,
                        "description": field.description,
                        "units": field.units,
                        "range": field.range,
                    }
                    for field in cmd_def.payload_schema.fields
                ],
                "total_length": cmd_def.payload_schema.total_length,
                "description": cmd_def.payload_schema.description,
                "custom_properties": cmd_def.payload_schema.custom_properties,
            },
            "decoder_instructions": cls._get_decoder_instructions(cmd_def.transport_envelope.payload_type),
        }
    
    @classmethod
    def _get_decoder_instructions(cls, payload_type: PayloadType) -> str:
        """Get deployment instructions for payload type."""
        instructions = {
            PayloadType.FLOAT_VECTOR: "Decode as array of floats, map to field names using payload schema",
            PayloadType.TEXT: "Decode as UTF-8 text, validate against grammar if provided",
            PayloadType.AUDIO: "Decode audio using sample_rate, channels, format from codec_info",
            PayloadType.IMAGE: "Decode image using width, height, channels from codec_info",
            PayloadType.PROTO: "Decode using protobuf definition in payload schema",
            PayloadType.BINARY: "Decode as raw bytes, structure defined in payload schema",
            PayloadType.JSON: "Parse as JSON, validate against schema",
            PayloadType.CUSTOM: "Use custom decoder defined in payload schema",
        }
        return instructions.get(payload_type, "Unknown payload type")
    
    # Backward compatibility methods for existing code
    @classmethod
    def get_all_command_types(cls) -> Dict[str, PyCommandTypeInfo]:
        """Backward compatibility: get old-style command type info."""
        result = {}
        for command_type, cmd_def in cls.get_all_commands().items():
            result[command_type] = PyCommandTypeInfo(
                command_type=cmd_def.command_type,
                description=cmd_def.payload_schema.description,
                fields=cmd_def.payload_schema.fields
            )
        return result
    
    @classmethod
    def get_command_type_info(cls, command_type: str) -> Optional[PyCommandTypeInfo]:
        """Backward compatibility: get command type info by name."""
        return cls.get_all_command_types().get(command_type)
    
    @classmethod
    def list_types(cls) -> List[str]:
        """Backward compatibility: list all registered command types."""
        return cls.list_command_types()


# Utility functions for creating metadata
def create_joint_biases_from_training_data(
    joint_biases_input,
    training_data: Optional[Dict[str, Any]] = None
) -> List[PyJointBias]:
    """Create joint biases from various input formats.
    
    Args:
        joint_biases_input: Either:
            - List of (name, angle, weight) tuples (from kos-zbot/train.py style)
            - List of joint names (for default biases)
        training_data: Optional dictionary with joint bias data
    
    Returns:
        List of PyJointBias objects
    """
    # Handle tuple list format (from train.py JOINT_BIASES)
    if (isinstance(joint_biases_input, list) and 
        len(joint_biases_input) > 0 and 
        isinstance(joint_biases_input[0], tuple) and 
        len(joint_biases_input[0]) == 3):
        return [
            PyJointBias(joint_name=name, reference_angle=angle, weight=weight)
            for name, angle, weight in joint_biases_input
        ]
    
    # Handle joint names list format
    elif isinstance(joint_biases_input, list) and all(isinstance(x, str) for x in joint_biases_input):
        joint_biases = []
        for joint_name in joint_biases_input:
            # Default values
            reference_angle = 0.0
            weight = 1.0
            
            # Extract from training data if available
            if training_data and 'joint_biases' in training_data:
                joint_data = training_data['joint_biases'].get(joint_name, {})
                reference_angle = joint_data.get('reference_angle', reference_angle)
                weight = joint_data.get('weight', weight)
            
            joint_biases.append(PyJointBias(joint_name, reference_angle, weight))
        
        return joint_biases
    
    # Invalid input
    else:
        raise ValueError(f"Invalid input format. Expected list of (name, angle, weight) tuples or list of joint names, got: {type(joint_biases_input)}")


def create_command_field(name: str, description: str, units: Optional[str] = None, 
                        range_tuple: Optional[tuple] = None) -> PyCommandField:
    """Utility function to create command fields with consistent formatting."""
    return PyCommandField(
        name=name,
        description=description,
        units=units,
        range=range_tuple
    )


def validate_payload_data(payload_data: Any, command_definition: CommandDefinition) -> bool:
    """Validate payload data against command definition."""
    envelope = command_definition.transport_envelope
    schema = command_definition.payload_schema
    
    if envelope.payload_type == PayloadType.FLOAT_VECTOR:
        if not isinstance(payload_data, (list, tuple)):
            return False
        if envelope.payload_length and len(payload_data) != envelope.payload_length:
            return False
        return all(isinstance(x, (int, float)) for x in payload_data)
    
    elif envelope.payload_type == PayloadType.TEXT:
        if not isinstance(payload_data, str):
            return False
        if envelope.codec_info and 'max_length' in envelope.codec_info:
            return len(payload_data) <= envelope.codec_info['max_length']
        return True
    
    elif envelope.payload_type == PayloadType.JSON:
        try:
            import json
            if isinstance(payload_data, str):
                json.loads(payload_data)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    # For other types (AUDIO, IMAGE, etc.), basic existence check
    return payload_data is not None


def get_example_payload(command_type: str, registry_class=None) -> Optional[Any]:
    """Generate example payload for a command type."""
    if registry_class is None:
        # Import here to avoid circular imports
        from kinfer.command_registry import CommandRegistry
        registry_class = CommandRegistry
    
    cmd_def = registry_class.get_command_definition(command_type)
    if not cmd_def:
        return None
    
    envelope = cmd_def.transport_envelope
    schema = cmd_def.payload_schema
    
    if envelope.payload_type == PayloadType.FLOAT_VECTOR:
        # Generate example values from field ranges
        example = []
        for field in schema.fields:
            if field.range:
                # Use midpoint of range
                min_val, max_val = field.range
                example.append((min_val + max_val) / 2)
            else:
                example.append(0.0)
        return example
    
    elif envelope.payload_type == PayloadType.TEXT:
        return "walk forward"
    
    elif envelope.payload_type == PayloadType.JSON:
        return '{"command": "example"}'
    
    elif envelope.payload_type == PayloadType.CUSTOM:
        if cmd_def.command_type == "pose":
            return None  # No input needed for pose
    
    return None