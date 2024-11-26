import platform

if platform.system() in ("Linux", "Windows"):
    from .tensorrt import *
