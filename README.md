# kinfer

This package is designed to support exporting and running inference on PyTorch models.

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
