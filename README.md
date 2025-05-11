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

To install `onnxruntime`, you can use pip:

```bash
pip install 'onnxruntime==1.20.0'
```

After doing this, you need to set `ORT_DYLIB_PATH` to point to the dynamic library.

```bash
python -c 'import onnxruntime as ort ; from pathlib import Path ; print(next((Path(ort.__file__).parent / "capi").glob("*.dylib")))'
```

Make sure this file actually exists!
