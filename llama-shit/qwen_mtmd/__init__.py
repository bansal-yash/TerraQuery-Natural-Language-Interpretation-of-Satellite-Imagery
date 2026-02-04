"""Package shim that re-exports the compiled qwen_mtmd binding."""

from . import qwen_mtmd as _core  # type: ignore[import]
from .qwen_mtmd import *  # re-export every binding symbol

__all__ = getattr(_core, "__all__", [name for name in dir() if not name.startswith("_")])
