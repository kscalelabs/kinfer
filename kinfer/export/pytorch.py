"""PyTorch model export utilities."""

import inspect
from dataclasses import fields, is_dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
import onnxruntime as ort
import torch
from torch import nn


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Extract model information including input parameters and their types.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary containing model information
    """
    # Get model's forward method signature
    signature = inspect.signature(model.forward)

    # Extract parameter information
    params_info = {}
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        params_info[name] = {
            'annotation': str(param.annotation),
            'default': None if param.default is param.empty else str(param.default)
        }

    return {
        'input_params': params_info,
        'num_parameters': sum(p.numel() for p in model.parameters()),
    }

def add_metadata_to_onnx(
    model_proto: onnx.ModelProto,
    metadata: Dict[str, Any],
    config: Optional[object] = None
) -> onnx.ModelProto:
    """Add metadata to ONNX model.

    Args:
        model_proto: ONNX model prototype
        metadata: Dictionary of metadata to add
        config: Optional configuration dataclass to add to metadata

    Returns:
        ONNX model with added metadata
    """
    # Add model metadata
    for key, value in metadata.items():
        meta = model_proto.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    # Add configuration if provided
    if config is not None and is_dataclass(config):
        for field in fields(config):
            value = getattr(config, field.name)
            meta = model_proto.metadata_props.add()
            meta.key = field.name
            meta.value = str(value)

    return model_proto

def infer_input_shapes(model: nn.Module) -> Union[torch.Size, List[torch.Size]]:
    """Infer input shapes from model architecture.

    Args:
        model: PyTorch model to analyze

    Returns:
        Single input shape or list of input shapes
    """
    # Try to find first layer
    first_layer = None

    # Check if model is Sequential
    if isinstance(model, nn.Sequential):
        first_layer = model[0]
    # Check if model has named modules
    else:
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                first_layer = module
                break

    if first_layer is None:
        raise ValueError("Could not determine input shape from model architecture")

    # Get input dimensions
    if isinstance(first_layer, nn.Linear):
        return torch.Size([1, first_layer.in_features])
    elif isinstance(first_layer, nn.Conv1d):
        raise ValueError("Cannot infer sequence length for Conv1d layer. Please provide input_tensors explicitly.")
    elif isinstance(first_layer, nn.Conv2d):
        raise ValueError("Cannot infer image dimensions for Conv2d layer. Please provide input_tensors explicitly.")
    elif isinstance(first_layer, nn.Conv3d):
        raise ValueError("Cannot infer volume dimensions for Conv3d layer. Please provide input_tensors explicitly.")
    else:
        raise ValueError(f"Unsupported layer type: {type(first_layer)}")

def create_example_inputs(model: nn.Module) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Create example input tensors based on model's forward signature and architecture.

    Args:
        model: PyTorch model to analyze

    Returns:
        Single tensor or tuple of tensors matching the model's expected input
    """
    signature = inspect.signature(model.forward)
    params = [p for p in signature.parameters.items() if p[0] != 'self']

    # If single parameter (besides self), try to infer shape
    if len(params) == 1:
        shape = infer_input_shapes(model)
        return torch.randn(*shape) if isinstance(shape, torch.Size) else torch.randn(*shape[0])


    # For multiple parameters, try to infer from parameter annotations
    input_tensors = []
    for name, param in params:
        # Try to get shape from annotation
        if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is torch.Tensor:
            # If annotation includes size information (e.g., Tensor[batch_size, channels, height, width])
            if hasattr(param.annotation, '__args__'):
                shape = param.annotation.__args__
                input_tensors.append(torch.randn(*shape) if isinstance(shape, torch.Size) else torch.randn(*shape[0]))
            else:
                # Default to a vector if no size info
                input_tensors.append(torch.randn(1, 32))
        else:
            # Default fallback
            input_tensors.append(torch.randn(1, 32))

    return tuple(input_tensors)

def export_to_onnx(
    model: nn.Module,
    input_tensors: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    config: Optional[object] = None,
    save_path: Optional[str] = None,
) -> ort.InferenceSession:
    """Export PyTorch model to ONNX format with metadata.

    Args:
        model: PyTorch model to export
        input_tensors: Optional example input tensors for model tracing. If None, will attempt to infer.
        config: Optional configuration dataclass to add to metadata
        save_path: Optional path to save the ONNX model

    Returns:
        ONNX inference session
    """
    # Get model information
    model_info = get_model_info(model)

    # Create example inputs if not provided
    if input_tensors is None:
        try:
            input_tensors = create_example_inputs(model)
            model_info['inferred_input_shapes'] = str(
                input_tensors.shape if isinstance(input_tensors, torch.Tensor)
                else [t.shape for t in input_tensors]
            )
        except ValueError as e:
            raise ValueError(f"Could not automatically infer input shapes. Please provide input_tensors. Error: {str(e)}")

    # Convert model to JIT if not already
    if not isinstance(model, torch.jit.ScriptModule):
        model = torch.jit.script(model)

    # Export model to buffer
    buffer = BytesIO()
    torch.onnx.export(
        model,
        (input_tensors,) if isinstance(input_tensors, torch.Tensor) else input_tensors, 
        buffer
    )
    buffer.seek(0)

    # Load as ONNX model
    model_proto = onnx.load_model(buffer)

    # Add metadata
    model_proto = add_metadata_to_onnx(model_proto, model_info, config)

    # Save if path provided
    if save_path:
        onnx.save_model(model_proto, save_path)

    # Convert to inference session
    buffer = BytesIO()
    onnx.save_model(model_proto, buffer)
    buffer.seek(0)

    return ort.InferenceSession(buffer.read())