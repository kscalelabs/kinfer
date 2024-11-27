"""Post-installation script."""

import importlib
import logging
import platform
import sys
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_platform_specific_modules() -> None:
    """Check and inform about platform-specific module availability."""
    platform_name = platform.system().lower()
    machine = platform.machine().lower()

    if platform_name == "darwin":
        logger.info("macOS detected - TensorRT support is not available")
        return

    # Special handling for Tegra platforms
    if platform_name == "linux" and machine == "aarch64":
        from kinfer.optimize.tensorrt_build import is_tegra_platform
        if is_tegra_platform():
            logger.info("Tegra platform detected - Please install TensorRT using: sudo apt-get install tensorrt")
            return

    # Define platform-specific modules
    platform_modules = {
        "darwin": [],  # macOS has no platform-specific modules
        "linux": ["kinfer.optimize.tensorrt"],
        "windows": ["kinfer.optimize.tensorrt"],
    }

    expected_modules = platform_modules.get(platform_name, [])
    missing_modules: List[str] = []
    unsupported_modules: List[str] = []

    # Check for modules that should be available on this platform
    for module_name in expected_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)

    # Check for modules that aren't available due to platform
    all_possible_modules = set()
    for modules in platform_modules.values():
        all_possible_modules.update(modules)

    unsupported_modules = [
        module for module in all_possible_modules 
        if module not in platform_modules.get(platform_name, [])
    ]

    if unsupported_modules:
        logger.info("The following modules are not available on %s: %s", 
                   platform_name, ", ".join(unsupported_modules))

    if missing_modules:
        logger.warning("The following modules should be available on %s but were not found: %s. "
                      "This might indicate an incomplete installation.",
                      platform_name, ", ".join(missing_modules))

if __name__ == "__main__":
    try:
        check_platform_specific_modules()
    except Exception as e:
        logger.warning("Platform check encountered an error: %s", e)
        # Don't raise the error - we don't want to fail the installation
        print("WARNING: Platform check encountered an error: %s", e)
        sys.exit(0)
