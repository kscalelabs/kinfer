import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tensorrt import optimize_model

if platform.system() in ("Linux", "Windows"):
    try:
        from .tensorrt import *
    except ImportError:
        # TensorRT not available, but that's okay
        pass
