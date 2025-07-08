# Kinfer Metadata Architecture

## Core Concepts

### Transport Envelope vs Payload Schema

The kinfer metadata system separates two fundamental concerns for deployment teams:

**Transport Envelope** answers: *"How do I encode this command for the runtime?"*
- What format should I encode to? (float32, UTF-8, JPEG, etc.)
- What's the expected length? (fixed/variable)
- What codec parameters do I use? (endianness, sample rate, etc.)

**Payload Schema** answers: *"What command should I construct?"*
- What fields are required?
- What units should I use? (m/s, radians, Hz)
- What are the valid ranges?
- What's the semantic meaning?

### Example: Float Vector Command

```python
# Transport Envelope - HOW to encode
TransportEnvelope(
    payload_type=PayloadType.FLOAT_VECTOR,
    payload_length=6,  # Always exactly 6 floats
    codec_info={"format": "float32", "endianness": "little"}
)

# Payload Schema - WHAT to construct
PayloadSchema(
    fields=[
        PyCommandField(name="vx", description="Forward velocity", units="m/s", range=(-2.0, 2.0)),
        PyCommandField(name="vy", description="Lateral velocity", units="m/s", range=(-1.0, 1.0)),
        # ... etc
    ],
    description="Unified command interface for humanoid robot locomotion"
)
```

## Payload Types & Encoding

### FLOAT_VECTOR
- **Use case**: Numerical control commands
- **Encoder**: Pack as array of floats using struct format
- **Validation**: Check field count, range constraints
- **Examples**: `unified_command_v1`, `joystick_walk_v1`

```python
# Encoding example
command_values = [1.5, 0.0, 0.1, 0.02, 0.0, -0.05]
encoded_bytes = struct.pack('<6f', *command_values)  # little-endian float32
```

### TEXT
- **Use case**: Natural language commands
- **Encoder**: UTF-8 string encoding
- **Validation**: Grammar rules, vocabulary, length limits
- **Examples**: `text_command_v1`

```python
# Encoding example
text_command = "walk forward slowly"
encoded_bytes = text_command.encode('utf-8')
```

### AUDIO
- **Use case**: Voice commands
- **Encoder**: Audio format specific (WAV, MP3, etc.)
- **Validation**: Duration, sample rate, channels
- **Examples**: `voice_command_v1`

```python
# Encoding example
with open('voice_command.wav', 'rb') as f:
    encoded_bytes = f.read()
# Validate: duration < 5000ms, sample_rate == 16000, channels == 1
```

### IMAGE
- **Use case**: Visual input commands
- **Encoder**: Image format specific (JPEG, PNG, etc.)
- **Validation**: Dimensions, color channels, file size
- **Examples**: `vision_navigation_v1`

```python
# Encoding example
import cv2
image = cv2.imread('camera_frame.jpg')
_, encoded_bytes = cv2.imencode('.jpg', image)
```

### JSON
- **Use case**: Structured multi-modal data
- **Encoder**: JSON serialization + UTF-8 encoding
- **Validation**: JSON schema compliance
- **Examples**: `multimodal_conversation_v1`

```python
# Encoding example
command_data = {
    "text_input": "go to kitchen",
    "audio_input": base64_encoded_audio,
    "confidence_scores": {"text_confidence": 0.95, "audio_confidence": 0.87}
}
encoded_bytes = json.dumps(command_data).encode('utf-8')
```

### CUSTOM
- **Use case**: Special purpose or no-command models
- **Encoder**: Custom logic defined in schema
- **Validation**: Custom validation rules
- **Examples**: `no_command`

```python
# Encoding example (no_command)
encoded_bytes = b""  # Empty payload for no-command models
```

## Registry Pattern for Deployment

```python
class CommandTypeRegistry:
    """Base class for command registries"""
    
    @classmethod
    def get_all_commands(cls) -> Dict[str, CommandDefinition]:
        """Override in subclasses to define available commands"""
        raise NotImplementedError
        
    @classmethod
    def get_encoder_info(cls, command_type: str) -> Dict:
        """Get deployment-ready encoder information"""
        # Returns complete encoding instructions

# Deployment team workflow:
def send_robot_command(command_type: str, **kwargs):
    # 1. Get command definition
    cmd_def = CommandRegistry.get_command_definition(command_type)
    
    # 2. Validate against schema
    validate_command_fields(kwargs, cmd_def.payload_schema)
    
    # 3. Encode according to transport envelope
    encoded_payload = encode_payload(kwargs, cmd_def.transport_envelope)
    
    # 4. Send to kinfer runtime
    kinfer_runtime.send_command(encoded_payload)
```

### Registry Benefits
- **Extensibility**: Easy to add new command types
- **Validation**: Built-in compatibility checking
- **Documentation**: Self-describing command definitions
- **Deployment**: Automatic encoder instruction generation

## Practical Encoding Examples

### Unified Command (6-DOF Locomotion)
```python
# High-level command construction
def send_walk_command(vx=0.0, vy=0.0, heading=0.0, body_height=0.0, roll=0.0, pitch=0.0):
    # 1. Validate ranges
    cmd_def = CommandRegistry.get_command_definition("unified_command_v1")
    validate_ranges([vx, vy, heading, body_height, roll, pitch], cmd_def.payload_schema)
    
    # 2. Encode as float32 array
    command_values = [vx, vy, heading, body_height, roll, pitch]
    encoded_bytes = struct.pack('<6f', *command_values)
    
    # 3. Send to kinfer runtime
    kinfer_runtime.send_command(encoded_bytes)

# Usage
send_walk_command(vx=1.5, vy=0.0, heading=0.1, body_height=0.02, pitch=-0.05)
```

### Joystick Command (One-hot + Frequency)
```python
def send_joystick_command(action="stand_still", gait_frequency=1.4):
    # 1. Convert action to one-hot
    actions = ["stand_still", "move_forward", "move_backward", "strafe_left", 
               "strafe_right", "turn_left", "turn_right"]
    one_hot = [1.0 if actions[i] == action else 0.0 for i in range(7)]
    
    # 2. Combine with frequency
    command_values = one_hot + [gait_frequency]
    
    # 3. Encode and send
    encoded_bytes = struct.pack('<8f', *command_values)
    kinfer_runtime.send_command(encoded_bytes)

# Usage
send_joystick_command("move_forward", gait_frequency=1.6)
```

### Text Command
```python
def send_text_command(text: str):
    # 1. Get transport envelope
    cmd_def = CommandRegistry.get_command_definition("text_command_v1")
    max_length = cmd_def.transport_envelope.codec_info["max_length"]
    
    # 2. Validate length
    if len(text.encode('utf-8')) > max_length:
        raise ValueError(f"Command too long (max {max_length} bytes)")
    
    # 3. Encode and send
    encoded_bytes = text.encode('utf-8')
    kinfer_runtime.send_command(encoded_bytes)

# Usage
send_text_command("walk forward slowly")
```

### Multi-Modal Command
```python
def send_multimodal_command(text_input=None, audio_data=None, fusion_weights=None):
    # 1. Construct JSON payload
    payload = {}
    if text_input:
        payload["text_input"] = text_input
    if audio_data:
        payload["audio_input"] = base64.b64encode(audio_data).decode('utf-8')
    if fusion_weights:
        payload["modality_weights"] = fusion_weights
    
    # 2. Validate against schema
    cmd_def = CommandRegistry.get_command_definition("multimodal_conversation_v1")
    validate_json_schema(payload, cmd_def.payload_schema.custom_properties["json_schema"])
    
    # 3. Encode and send
    encoded_bytes = json.dumps(payload).encode('utf-8')
    kinfer_runtime.send_command(encoded_bytes)

# Usage
send_multimodal_command(
    text_input="go to kitchen",
    audio_data=recorded_audio_bytes,
    fusion_weights={"text_confidence": 0.95, "audio_confidence": 0.87}
)
```

## Error Handling Strategy

When deployment encounters unknown command types:

1. **Extract transport envelope** from model metadata
2. **Check registry** for command type
3. **If unknown**: Print complete encoder instructions
4. **Include**: Payload schema, field definitions, examples
5. **Actionable**: Deployment team can implement encoder

```python
# Example error handling
try:
    send_robot_command("experimental_gait_v2", step_length=0.5)
except UnknownCommandType as e:
    print(f"Unknown command type: {e.command_type}")
    print("Encoder instructions:")
    print(f"  Transport: {e.transport_envelope}")
    print(f"  Schema: {e.payload_schema}")
    print(f"  Example: {e.example_payload}")
```

## Integration with Kinfer Runtime

The kinfer runtime receives encoded commands and:

1. **Decodes** according to transport envelope
2. **Validates** against payload schema
3. **Executes** the command in the policy
4. **Returns** action outputs

```python
# Kinfer runtime perspective (for reference)
class KinferRuntime:
    def receive_command(self, encoded_bytes: bytes):
        # 1. Decode according to transport envelope
        command = self.decode_command(encoded_bytes)
        
        # 2. Validate against payload schema
        if not self.validate_command(command):
            raise InvalidCommandError()
        
        # 3. Execute in policy
        action = self.policy.step(command, self.internal_state)
        
        return action
```

This architecture enables robust, extensible command interfaces between deployment systems and kinfer policy runtimes, with clear separation between transport concerns and semantic meaning.
