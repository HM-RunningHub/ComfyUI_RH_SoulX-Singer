from . import _compat  # noqa: F401  (dependency shims must run before the model code is imported)
from .nodes import NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {k:k for k,v in NODE_CLASS_MAPPINGS.items()}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']