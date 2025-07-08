## K-Infer: KScale Inference Runtime

**Kinfer** is a complete policy runtime environment that enables trained JAX policies to be exported to portable, self-describing deployment packages and executed in simulation and real-time controller on robots.

For more information, see the documentation [here](https://docs.kscale.dev/docs/k-infer).

To enable logging, set your log path in the environment variable `KINFER_LOG_PATH`. For example,
```
export KINFER_LOG_PATH=/home/dpsh/kinfer-logs
```

## Core Workflow
```
JAX Training → convert.py → ONNX Runtime → .kinfer Package
```

## What Kinfer Provides

### 1. **JAX-to-Portable Export Pipeline**
- **Input**: Trained JAX models (like `KbotWalkingTask`)
- **Export**: `init_fn` and `step_fn` functions via `export_fn`
- **Runtime**: ONNX-based inference engine
- **Output**: Self-contained `.kinfer` packages

### 2. **Metadata System**
- **Joint Biases**: Reference angles and weights for each joint
- **Command Type Information**: Structured command definitions  
- **Version Tracking**: Kinfer version compatibility
- **Training Metadata**: Arbitrary training-time data

### 3. **Command/Payload/Transport Architecture**
- **Transport Envelope**: HOW to decode data (codec, format, length)
- **Payload Schema**: WHAT the decoded data means (fields, units, ranges)
- **Command Registry**: Centralized command type definitions

### 4. **Polyglot Release Structure**
- Runtime packages carry their own source code
- Self-describing model files

## Key Benefits

The kinfer metadata system provides a robust framework for defining, transporting, and deploying robot command interfaces. It separates **transport concerns** (how data is encoded) from **semantic concerns** (what the data means).

### Example: JAX Policy → Kinfer Package

```python
# convert.py: Export trained JAX policy
@jax.jit
def step_fn(joint_angles, joint_velocities, quaternion, heading, command, carry):
    # JAX policy logic here
    obs = jnp.concatenate([joint_angles, joint_velocities, ...])
    dist, carry = model.actor.forward(obs, carry)
    return dist.mode(), carry

# Export with rich metadata
metadata = PyModelMetadata(
    joint_names=joint_names,
    num_commands=NUM_COMMANDS,
    carry_size=carry_shape,
    command_type_info=command_info,  # ← Command structure
    joint_biases=joint_biases,       # ← Training-derived info
    kinfer_version="1.0"
)

# Package everything into portable runtime
init_onnx = export_fn(init_fn, metadata)
step_onnx = export_fn(step_fn, metadata)
kinfer_model = pack(init_onnx, step_onnx, metadata)
```

### Deployment Advantages
- **Portable**: Works across languages/platforms via ONNX
- **Self-describing**: Command interfaces defined in metadata
- **Robust**: Rich validation and error handling
- **Extensible**: Easy to add new command types

Kinfer is the complete bridge from "trained JAX policy" to "deployable robot runtime" - with the metadata and command architecture being key enablers of that portability and robustness.

## Quick Start

```bash
# List all available command types
kinfer commands

# Show details for a specific command
kinfer commands unified_command_v1 --details

# Inspect a model file
kinfer inspect my_model.kinfer

# Check transport envelope
kinfer envelope my_model.kinfer
```

## Command Types Overview

| Command Type | Payload Type | Description |
|--------------|--------------|-------------|
| `no_command` | CUSTOM | Models that operate without external commands |
| `unified_command_v1` | FLOAT_VECTOR | Standard 6-DOF humanoid locomotion control |
| `joystick_walk_v1` | FLOAT_VECTOR | Discrete joystick interface + gait frequency |
| `classical_tilt_v1` | FLOAT_VECTOR | IMU-based tilt compensation parameters |
| `voice_command_v1` | AUDIO | Natural language voice commands |
| `text_command_v1` | TEXT | Natural language text commands |
| `vision_navigation_v1` | IMAGE | Camera-based navigation |
| `multimodal_conversation_v1` | JSON | Combined audio + text interaction |

## Architecture
```
Command Type ("unified_command_v1")
├── Transport Envelope (HOW to decode)
│ ├── payload_type: FLOAT_VECTOR
│ ├── payload_length: 6
│ └── codec_info: {format: "float32", endianness: "little"}
└── Payload Schema (WHAT it means)
├── fields: [vx, vy, heading, body_height, roll, pitch]
├── units: ["m/s", "m/s", "radians", "m", "radians", "radians"]
└── ranges: [(-2,2), (-1,1), (-π,π), (-0.1,0.1), (-0.3,0.3), (-0.3,0.3)]
```


k-infer portable runtime file format
---------------------------------------------------------------------------------- 
k-infer runtime file is a polyglot ( model and comopressed archive ) meaning it is simultaneously a kinfer model and a ZIP file.
The ZIP file contains the original training and export code, checkpoint, config and metadata

The first 32bytes contain the ZIP local header, along with some magic, then the actual kinfer model followed by a compressed archive

```
┌──────────  32 B  ──────────┐
│ 0x0000  ZIP local-header #0  ← overlaps kinfer “front porch” (30 B)
│ 0x001E  0xCE  kinfer-polyglot magic
│ 0x001F  0x01  kinfer hdr-version / flags
└────────────────────────────┘
0x0020  ──── the original kinfer header & payload ────  (LEN = K)
          .
          .                   ← kinfer reader stops here (0x0020+K)
          .
┌──────────── additional ZIP entries (optional) ────────────────────────┐
│ 0x0020+K  local-hdr #1 (training_code.py”), data                      |
| -         local-hdr #1 (convert.py”), data                            │
| -         local-hdr #1 (joint_config_table.txt”), data                │
| -         local-hdr #1 (config.yaml”), data                           │
| -         local-hdr #1 (info.json”), data                             │
| -         local-hdr #1 (logs.txt”), data                              │
| -         local-hdr #1 (state.txt”), data                             │
| -         local-hdr #1 (checkpoints/*”), data                         │
│ …                                                                     │
│ local-hdr #N (“assets/img.png”), data                                 │
└───────────────────────────────────────────────────────────────────────┘
central-directory (1 + N records)   ← points back to *all* local hdrs  
EOCD
```