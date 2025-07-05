"""Specific command definitions for deployment use."""

from typing import Dict, List, Optional, Any
from kinfer.commands import (
    CommandTypeRegistry, CommandDefinition, TransportEnvelope, PayloadSchema,
    PayloadType, PyCommandField
)


class DeploymentCommandRegistry(CommandTypeRegistry):
    """Production registry with all supported command types."""
    
    @classmethod
    def get_all_commands(cls) -> Dict[str, CommandDefinition]:
        """Get all registered command definitions."""
        return {
            # Float vector commands
            "unified_command_v1": cls.get_unified_command_v1(),
            "joystick_walk_v1": cls.get_joystick_walk_v1(),
            "classical_tilt_v1": cls.get_classical_tilt_v1(),
            
            # Simple commands
            "pose": cls.get_pose(),
            
            # Multi-modal commands
            "voice_command_v1": cls.get_voice_command_v1(),
            "text_command_v1": cls.get_text_command_v1(),
            "vision_navigation_v1": cls.get_vision_navigation_v1(),
            "multimodal_conversation_v1": cls.get_multimodal_conversation_v1(),
        }
    
    # === FLOAT VECTOR COMMANDS ===
    
    @staticmethod
    def get_unified_command_v1() -> CommandDefinition:
        """Unified command type for humanoid robot control."""
        fields = [
            PyCommandField(
                name="vx",
                description="Forward velocity command",
                units="m/s",
                range=(-2.0, 2.0)
            ),
            PyCommandField(
                name="vy", 
                description="Lateral velocity command",
                units="m/s",
                range=(-1.0, 1.0)
            ),
            PyCommandField(
                name="heading",
                description="Absolute heading target",
                units="radians",
                range=(-3.14159, 3.14159)
            ),
            PyCommandField(
                name="body_height",
                description="Relative body height adjustment",
                units="m",
                range=(-0.1, 0.1)
            ),
            PyCommandField(
                name="roll",
                description="Body roll angle",
                units="radians", 
                range=(-0.3, 0.3)
            ),
            PyCommandField(
                name="pitch",
                description="Body pitch angle",
                units="radians",
                range=(-0.3, 0.3)
            ),
        ]
        
        return CommandDefinition(
            command_type="unified_command_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.FLOAT_VECTOR,
                payload_length=6,
                codec_info={"format": "float32", "endianness": "little"}
            ),
            payload_schema=PayloadSchema(
                fields=fields,
                total_length=6,
                description="Unified command interface for humanoid robot locomotion control. Supports walking, turning, and body pose adjustments."
            )
        )
    
    @staticmethod
    def get_joystick_walk_v1() -> CommandDefinition:
        """Discrete joystick walking command + gait frequency."""
        fields = [
            PyCommandField(
                name="stand_still",
                description="Robot should remain stationary",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="move_forward",
                description="Translate +X in robot frame",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="move_backward",
                description="Translate −X in robot frame",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="strafe_left",
                description="Translate +Y in robot frame",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="strafe_right",
                description="Translate −Y in robot frame",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="turn_left",
                description="Rotate +Z (CCW) in place",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="turn_right",
                description="Rotate −Z (CW) in place",
                units="boolean",
                range=(0.0, 1.0),
            ),
            PyCommandField(
                name="gait_frequency",
                description="Step frequency target",
                units="Hz",
                range=(1.0, 2.0),   # training range 1.2–1.5 Hz
            ),
        ]
        
        return CommandDefinition(
            command_type="joystick_walk_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.FLOAT_VECTOR,
                payload_length=8,
                codec_info={"format": "float32", "endianness": "little"}
            ),
            payload_schema=PayloadSchema(
                fields=fields,
                total_length=8,
                description="One-hot joystick interface for humanoid locomotion plus a scalar gait-frequency target. Exactly one of the first seven fields should be 1.0 at any tick (stand_still, forward, backward, strafe_left, strafe_right, turn_left, turn_right); all others 0.0. The eighth field sets the desired gait frequency in Hz."
            )
        )
    
    @staticmethod
    def get_classical_tilt_v1() -> CommandDefinition:
        """Normalized gain/scale tweaks for the IMU-based tilt controller."""
        fields = [
            PyCommandField(
                name="pitch_gain_adj",
                description="Normalized adjustment for pitch compensation gain",
                units="dimensionless",
                range=(-1.0, 1.0),
            ),
            PyCommandField(
                name="roll_gain_adj",
                description="Normalized adjustment for roll compensation gain",
                units="dimensionless",
                range=(-1.0, 1.0),
            ),
            PyCommandField(
                name="hip_pitch_scale_adj",
                description="Adjustment for hip-pitch scaling in tilt compensation",
                units="dimensionless",
                range=(-1.0, 1.0),
            ),
            PyCommandField(
                name="ankle_roll_scale_adj",
                description="Adjustment for ankle-roll scaling in tilt compensation",
                units="dimensionless",
                range=(-1.0, 1.0),
            ),
            PyCommandField(
                name="ankle_pitch_scale_adj",
                description="Adjustment for ankle-pitch scaling in tilt compensation",
                units="dimensionless",
                range=(-1.0, 1.0),
            ),
            PyCommandField(
                name="knee_pitch_scale_adj",
                description="Adjustment for knee-pitch scaling in tilt compensation",
                units="dimensionless",
                range=(-1.0, 1.0),
            ),
        ]
        
        return CommandDefinition(
            command_type="classical_tilt_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.FLOAT_VECTOR,
                payload_length=6,
                codec_info={"format": "float32", "endianness": "little"}
            ),
            payload_schema=PayloadSchema(
                fields=fields,
                total_length=6,
                description="""Six continuous parameters in the range [-1, 1] that modulate the gains and joint-specific scale factors used by the IMU tilt-compensation model.

Internally they map linearly onto:
  • pitch_gain          (2.0 → 3.5)  
  • roll_gain           (1.5 → 2.5)  
  • hip_pitch_scale     (0.5 → 0.8)  
  • ankle_roll_scale    (0.5 → 0.8)  
  • ankle_pitch_scale   (0.5 → 0.8)  
  • knee_pitch_scale    (0.8 → 1.1)
where −1.0 reproduces the baseline values shown above and +1.0 applies the maximum adjustment."""
            )
        )
    
    # === SIMPLE COMMANDS ===
    
    @staticmethod
    def get_pose() -> CommandDefinition:
        """Simple pose command - no dynamic input."""
        return CommandDefinition(
            command_type="pose",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.CUSTOM,
                payload_length=0,
                codec_info={"custom_type": "pose"}
            ),
            payload_schema=PayloadSchema(
                fields=[],
                total_length=0,
                description="Simple pose command - no dynamic commands, just joint positions. Used for classical control policies that output static poses.",
                custom_properties={"type": "static_pose"}
            )
        )
    
    # === MULTI-MODAL COMMANDS ===
    
    @staticmethod
    def get_voice_command_v1() -> CommandDefinition:
        """Voice command for natural language control."""
        return CommandDefinition(
            command_type="voice_command_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.AUDIO,
                payload_length=None,  # Variable length
                codec_info={
                    "format": "wav",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "max_duration_ms": 5000,
                    "encoding": "pcm_s16le"
                }
            ),
            payload_schema=PayloadSchema(
                fields=[],  # No structured fields for audio
                total_length=0,
                description="Voice commands for natural language robot control",
                custom_properties={
                    "supported_languages": ["en-US"],
                    "wake_words": ["robot", "zbot", "hey robot"],
                    "command_vocabulary": [
                        "walk", "run", "stop", "turn", "sit", "stand",
                        "forward", "backward", "left", "right", "faster", "slower"
                    ],
                    "confidence_threshold": 0.8,
                    "noise_cancellation": True
                }
            )
        )
    
    @staticmethod
    def get_text_command_v1() -> CommandDefinition:
        """Text command for natural language control."""
        return CommandDefinition(
            command_type="text_command_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.TEXT,
                payload_length=None,  # Variable length
                codec_info={
                    "encoding": "utf-8",
                    "max_length": 1024,
                }
            ),
            payload_schema=PayloadSchema(
                fields=[],  # No structured fields for text
                total_length=0,
                description="Text commands for natural language robot control",
                custom_properties={
                    "grammar": "command ::= action object\naction ::= 'walk' | 'turn' | 'stop'\nobject ::= 'forward' | 'left' | 'right'",
                    "examples": ["walk forward", "turn left", "stop"],
                    "case_sensitive": False,
                    "punctuation_ignored": True
                }
            )
        )
    
    @staticmethod
    def get_vision_navigation_v1() -> CommandDefinition:
        """Vision-based navigation command."""
        return CommandDefinition(
            command_type="vision_navigation_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.IMAGE,
                payload_length=None,  # Variable length
                codec_info={
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "channels": 3,
                    "compression": "jpeg",
                    "max_size_bytes": 102400
                }
            ),
            payload_schema=PayloadSchema(
                fields=[],  # No structured fields for images
                total_length=0,
                description="Camera image for visual navigation commands",
                custom_properties={
                    "camera_type": "rgb_front",
                    "field_of_view": 90.0,
                    "processing_pipeline": "detect_obstacles_and_goals",
                    "supported_formats": ["jpeg", "png"],
                    "color_space": "sRGB"
                }
            )
        )
    
    @staticmethod
    def get_multimodal_conversation_v1() -> CommandDefinition:
        """Multi-modal conversation command."""
        fields = [
            PyCommandField(
                name="audio_data",
                description="Base64-encoded audio data",
                units="string",
                range=None
            ),
            PyCommandField(
                name="text_data",
                description="Text transcription or direct text input",
                units="string",
                range=None
            ),
            PyCommandField(
                name="modality_weights",
                description="Confidence weights for each modality",
                units="object",
                range=None
            )
        ]
        
        return CommandDefinition(
            command_type="multimodal_conversation_v1",
            transport_envelope=TransportEnvelope(
                payload_type=PayloadType.JSON,
                payload_length=None,  # Variable length
                codec_info={
                    "encoding": "utf-8",
                    "schema_version": "1.0",
                    "max_size_bytes": 1048576
                }
            ),
            payload_schema=PayloadSchema(
                fields=fields,
                total_length=0,
                description="Multi-modal conversation with audio and text inputs",
                custom_properties={
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "audio_data": {
                                "type": "string",
                                "description": "Base64-encoded WAV audio",
                                "format": "base64"
                            },
                            "text_data": {
                                "type": "string",
                                "maxLength": 1024,
                                "description": "Direct text input or transcription"
                            },
                            "modality_weights": {
                                "type": "object",
                                "properties": {
                                    "audio_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    "text_confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                },
                                "required": ["audio_confidence", "text_confidence"]
                            }
                        },
                        "anyOf": [
                            {"required": ["audio_data"]},
                            {"required": ["text_data"]},
                            {"required": ["audio_data", "text_data"]}
                        ]
                    },
                    "audio_config": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "bit_depth": 16,
                        "max_duration_ms": 10000
                    },
                    "fusion_strategy": "weighted_average",
                    "fallback_behavior": "prefer_text_on_audio_failure"
                }
            )
        )