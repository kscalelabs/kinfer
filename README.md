# kinfer

This package is designed to support running real-time robotics models.

## Basics

Each `kinfer` model is composed of two static neural network graphs, `init` and `step`. They are then run in the following control loop:

```python
state = init()
while True:
  model_input = {}
  for name in input_names:
    model_input[name] = get_model_input(name)
  model_output, state = step(model_input, state)
  do_model_output(model_output)
  sleep_until(next_step_time)
```

The `kinfer` client implements the functions `get_model_input` and `do_model_output`. The `input_names` are inferred from the `step` function graph. `kinfer` provides handshake warnings for certain common keys.

## Installation

```bash
pip install kinfer
```

### ONNX Runtime

You can install the latest version of ONNX Runtime on Mac with:

```bash
brew install onnxruntime
```

You may need to add the binary to your DYLD_LIBRARY_PATH:

```bash
$ brew ls onnxruntime
/opt/homebrew/Cellar/onnxruntime/1.20.1/include/onnxruntime/ (11 files)
/opt/homebrew/Cellar/onnxruntime/1.20.1/lib/libonnxruntime.1.20.1.dylib  # <-- This is the binary
/opt/homebrew/Cellar/onnxruntime/1.20.1/lib/cmake/ (4 files)
/opt/homebrew/Cellar/onnxruntime/1.20.1/lib/pkgconfig/libonnxruntime.pc
/opt/homebrew/Cellar/onnxruntime/1.20.1/lib/libonnxruntime.dylib
/opt/homebrew/Cellar/onnxruntime/1.20.1/sbom.spdx.json
$ export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/onnxruntime/1.20.1/lib:$DYLD_LIBRARY_PATH
```

### Considerations for Exporting PyTorch Models

Don't use common names for the inputs to your forward pass. E.g. `input`, `output`, `state`, `state_tensor`, `buffer`, etc.

This is because ONNX has internal names for the model and if there's a conflict, the inputs will have a .1, .2, etc. suffix which makes it really hard to figure out what value_name to pass into your kinfer io values.
