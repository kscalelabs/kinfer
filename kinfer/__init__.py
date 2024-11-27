__version__ = "0.0.4"

import sys
from importlib.util import find_spec

from kinfer.python import export, inference

sys.modules["kinfer.export"] = export
sys.modules["kinfer.inference"] = inference

__all__ = ["export", "inference", "__version__"]
