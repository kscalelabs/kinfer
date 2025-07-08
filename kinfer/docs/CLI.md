# Kinfer CLI Guide

## Installation & Setup

```bash
# Install kinfer with CLI support
pip install -e kinfer/

# Verify installation
kinfer --version
```

## Commands Overview

### `kinfer commands`
List all available command types with summary information.

```bash
# List all command types (summary view)
kinfer commands

# Output:
# Available Command Types:
# ===================================================
#   no_command           | CUSTOM       | Model operates without external commands.
#   unified_command_v1   | FLOAT_VECTOR | Unified command interface for humanoid robot locomotion control.
#   joystick_walk_v1     | FLOAT_VECTOR | One-hot joystick interface for humanoid locomotion plus gait frequency.
#   voice_command_v1     | AUDIO        | Voice commands for natural language robot control.
```

### `kinfer commands <command_name>`
Show detailed information about a specific command type.

```bash
# Basic command info
kinfer commands unified_command_v1

# Output:
# Command: unified_command_v1
# ===========================
# 
# Description:
#   Unified command interface for humanoid robot locomotion control.
# 
# Payload Type: FLOAT_VECTOR
# Fields: 6
#   • vx [m/s]
#   • vy [m/s]
#   • heading [radians]
#   • body_height [m]
#   • roll [radians]
#   • pitch [radians]
```

### `kinfer commands <command_name> --details`
Show complete implementation details for deployment.

```bash
# Complete command details
kinfer commands unified_command_v1 --details

# Output includes:
# - Full field descriptions and ranges
# - Transport envelope specifications
# - Decoder instructions
# - Custom properties
# - Validation rules
```

### `kinfer inspect <model.kinfer>`
Inspect a kinfer model file and display all metadata.

```bash
# Inspect model metadata
kinfer inspect my_robot_policy.kinfer

# Output:
# Kinfer Runtime Package Inspection: my_robot_policy.kinfer
# ============================================================
# 
# General Information:
#   Model Path: /path/to/my_robot_policy.kinfer
#   Kinfer Version: 1.0
#   File Size: 45.2 MB
# 
# Model Configuration:
#   Joint Count: 10
#   Commands: 6
#   Carry Size: [256, 256]
# 
# Joint Names:
#    1. left_hip_pitch
#    2. left_hip_roll
#    ...
# 
# Joint Biases:
#   left_hip_pitch: ref=0.100, weight=1.000
#   left_hip_roll: ref=-0.050, weight=0.800
#   ...
# 
# Command Type: unified_command_v1
# Description: Unified command interface for humanoid robot locomotion control.
# 
# Command type 'unified_command_v1' is registered in the registry
```

### `kinfer envelope <model.kinfer>`
Show transport envelope information for deployment.

```bash
# Check transport envelope
kinfer envelope my_robot_policy.kinfer

# Output:
# Transport Envelope: my_robot_policy.kinfer
# ================================================
# Command Type: unified_command_v1
# Payload Type: FLOAT_VECTOR
# Payload Length: 6
# Codec Info: {'format': 'float32', 'endianness': 'little'}
# 
# Decoder Instructions:
#   Decode as array of floats, map to field names using payload schema
```

## Practical Workflows

### Exploring Available Commands
```bash
# 1. See what's available
kinfer commands

# 2. Pick an interesting command
kinfer commands voice_command_v1

# 3. Get implementation details
kinfer commands voice_command_v1 --details
```

### Model Validation
```bash
# 1. Inspect the model
kinfer inspect my_model.kinfer

# 2. Check if command type is supported

# 3. Get deployment information
kinfer envelope my_model.kinfer
```

### Deployment Integration
```bash
# 1. Extract transport envelope
kinfer envelope model.kinfer > transport_info.txt

# 2. Use transport info to implement decoder
# 3. Validate against payload schema
```

## Error Scenarios

### Unknown Command Type
```bash
kinfer inspect model_with_new_command.kinfer

# Output includes:
# Command type 'experimental_gait_v2' is NOT registered in the registry
# Available command types: unified_command_v1, joystick_walk_v1, ...
```

### Missing Model File
```bash
kinfer inspect nonexistent.kinfer
# Error: Path "nonexistent.kinfer" does not exist.
```

### Corrupted Model
```bash
kinfer inspect corrupted.kinfer
# Error inspecting model: No metadata.json found in model file
```

## Deployment Considerations

### Error Messages
When deployment encounters your new command:
1. The transport envelope tells them HOW to decode
2. The payload schema tells them WHAT it means
3. Your description helps them implement it correctly

### Documentation Generation
Your command definition automatically generates:
- CLI help text
- Decoder instructions
- Field validation rules
- Example payloads

This makes it easy for deployment teams to implement support for your command type.
```

## 5. Examples Guide: `kinfer/docs/EXAMPLES.md`

```
kinfer/docs/EXAMPLES.md
# Kinfer Examples

## Basic Usage Examples

### Creating a Model with Metadata

```python
# In your convert.py script
from kinfer.commands import create_joint_biases_from_training_data
from kinfer.commands_registry import CommandRegistry
from kinfer.rust_bindings import PyModelMetadata

# Create joint biases from training data
joint_biases = create_joint_biases_from_training_data(JOINT_BIASES)

# Get command type info
command_info = CommandRegistry.get_command_type_info("unified_command_v1")

# Create metadata
metadata = PyModelMetadata(
    joint_names=joint_names,
    num_commands=6,
    carry_size=[256, 256],
    joint_biases=joint_biases,
    command_type_info=command_info,
    kinfer_version="1.0"
)

# Export model with metadata
export_fn(inference_fn, metadata, "my_model.kinfer")
```

### CLI Exploration Workflow

```bash
# 1. See what command types are available
$ kinfer commands
Available Command Types:
===================================================
  no_command           | CUSTOM       | Model operates without external commands.
  unified_command_v1   | FLOAT_VECTOR | Unified command interface for humanoid robot locomotion control.
  joystick_walk_v1     | FLOAT_VECTOR | One-hot joystick interface for humanoid locomotion plus gait frequency.

# 2. Examine a specific command type
$ kinfer commands unified_command_v1
Command: unified_command_v1
===========================

Description:
  Unified command interface for humanoid robot locomotion control.

Payload Type: FLOAT_VECTOR
Fields: 6
  • vx [m/s]
  • vy [m/s]
  • heading [radians]
  • body_height [m]
  • roll [radians]
  • pitch [radians]

💡 Use --details for full implementation details:
   kinfer commands unified_command_v1 --details

# 3. Get complete implementation details
$ kinfer commands unified_command_v1 --details
# ... detailed field information, ranges, decoder instructions ...
```

### Model Inspection

```bash
# Inspect a model file
$ kinfer inspect walking_policy.kinfer
Kinfer Runtime Package Inspection: walking_policy.kinfer
============================================================

General Information:
  Model Path: /home/user/models/walking_policy.kinfer
  Kinfer Version: 1.0
  File Size: 12.3 MB

Model Configuration:
  Joint Count: 10
  Commands: 6
  Carry Size: [128, 128]

Joint Names:
   1. left_hip_pitch
   2. left_hip_roll
   3. left_knee_pitch
   4. left_ankle_pitch
   5. left_ankle_roll
   6. right_hip_pitch
   7. right_hip_roll
   8. right_knee_pitch
   9. right_ankle_pitch
  10. right_ankle_roll

Joint Biases:
  left_hip_pitch: ref=0.100, weight=1.000
  left_hip_roll: ref=-0.050, weight=0.800
  left_knee_pitch: ref=0.200, weight=1.000
  # ... etc

Command Type: unified_command_v1
Description: Unified command interface for humanoid robot locomotion control.

Command Fields:
  • vx [m/s]: Forward velocity command (range: (-2.0, 2.0))
  • vy [m/s]: Lateral velocity command (range: (-1.0, 1.0))
  • heading [radians]: Absolute heading target (range: (-3.14159, 3.14159))
  • body_height [m]: Relative body height adjustment (range: (-0.1, 0.1))
  • roll [radians]: Body roll angle (range: (-0.3, 0.3))
  • pitch [radians]: Body pitch angle (range: (-0.3, 0.3))

  Command type 'unified_command_v1' is registered in the registry

Inspection completed successfully!
```

## Command Type Examples

### 1. No Command Model

```python
# For models that don't need external commands
@staticmethod
def get_no_command() -> CommandDefinition:
    return CommandDefinition(
        command_type="no_command",
        transport_envelope=TransportEnvelope(
            payload_type=PayloadType.CUSTOM,
            payload_length=0,
            codec_info={"custom_type": "no_command"}
        ),
        payload_schema=PayloadSchema(
            fields=[],
            total_length=0,
            description="Model operates without external commands. Policy generates actions based solely on sensor inputs and internal state.",
            custom_properties={
                "type": "no_command_policy",
                "requires_commands": False,
                "input_source": "sensors_only"
            }
        )
    )
```


### 2. Float Vector Commands

#### Unified Command (6-DOF)
```python
# 6-DOF locomotion control
command = [
    1.5,    # vx: forward velocity (m/s)
    0.0,    # vy: lateral velocity (m/s)  
    0.1,    # absolute heading
    0.02,   # body_height: lift body 2cm (m)
    0.0,    # roll: no body roll (radians)
    -0.05   # pitch: lean forward slightly (radians)
]
```

#### Joystick Command (One-hot + Frequency)
```python
# One-hot joystick interface
command = [
    0.0,  # stand_still
    1.0,  # move_forward (active)
    0.0,  # move_backward
    0.0,  # strafe_left
    0.0,  # strafe_right
    0.0,  # turn_left
    0.0,  # turn_right
    1.4   # gait_frequency (Hz)
]
```

### 3. Multi-Modal Commands

#### Voice Command
```python
# Audio payload with metadata
command_envelope = {
    "payload_type": "AUDIO",
    "codec_info": {
        "format": "wav",
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16
    }
}

# Audio data would be raw WAV bytes
# Deployment system uses codec_info to decode
```

#### Text Command
```python
# Simple text command
command = "walk forward slowly"

# Deployment system:
# 1. Decodes as UTF-8 text
# 2. Validates against grammar rules
# 3. Processes natural language
```

#### JSON Multi-Modal
```python
command = {
    "text_input": "go to the kitchen",
    "audio_input": "base64_encoded_wav_data_here",
    "confidence_scores": {
        "text_confidence": 0.95,
        "audio_confidence": 0.87
    }
}

# Deployment system:
# 1. Parses JSON
# 2. Validates against schema
# 3. Fuses multi-modal inputs
```

## Deployment Integration Examples

### Transport Envelope Inspection
```bash
$ kinfer envelope my_model.kinfer
Transport Envelope: my_model.kinfer
================================================
Command Type: unified_command_v1
Payload Type: FLOAT_VECTOR
Payload Length: 6
Codec Info: {'format': 'float32', 'endianness': 'little'}

Decoder Instructions:
  Decode as array of floats, map to field names using payload schema
```

### Error Handling for Unknown Commands
```bash
$ kinfer inspect model_with_new_command.kinfer
# ... normal inspection output ...

⚠ Command type 'experimental_gait_v2' is NOT registered in the registry

# Deployment team sees:
# 1. What the transport envelope expects
# 2. What the payload schema defines  
# 3. How to implement a decoder
# 4. Field definitions and validation rules
```

### Automated Model Validation
```python
# In deployment pipeline
def validate_model_compatibility(model_path):
    try:
        # Extract metadata
        metadata = extract_metadata(model_path)
        command_type = metadata.command_type_info.command_type
        
        # Check if we support this command type
        if CommandRegistry.validate_command_type(command_type):
            print(f"Model uses supported command type: {command_type}")
            return True
        else:
            print(f"Unknown command type: {command_type}")
            print(f"Available types: {CommandRegistry.list_command_types()}")
            
            # Get implementation instructions
            decoder_info = CommandRegistry.get_decoder_info(command_type)
            if decoder_info:
                print("Implementation required:")
                print(f"  Transport: {decoder_info['transport_envelope']}")
                print(f"  Schema: {decoder_info['payload_schema']}")
            
            return False
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False
```