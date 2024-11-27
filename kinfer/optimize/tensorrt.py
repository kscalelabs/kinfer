"""TensorRT optimization."""
import importlib.util
import warnings
from typing import Any, Optional


def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    return importlib.util.find_spec("tensorrt") is not None

if is_tensorrt_available():
    import tensorrt as trt

    def optimize_model() -> "trt.ICudaEngine":
        # Create TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)

        # Create builder and network
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Create config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        # Build and return engine
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        return engine
else:
    def optimize_model() -> "Any":  # type: ignore[ANN401]
        warnings.warn(
            "TensorRT is not available. Install kinfer[tensorrt] on a supported platform.",
            RuntimeWarning
        )
        return None
