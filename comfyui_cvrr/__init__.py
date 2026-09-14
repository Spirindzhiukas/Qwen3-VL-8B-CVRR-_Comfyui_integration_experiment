"""comfyui_cvrr -- CVRR (Causal Visual Recurrent Reasoning) as a ComfyUI text encoder.

Experimental integration of ``dmis-lab/Qwen3-VL-8B-CVRR`` with FLUX.2 Klein
image editing.  See ``docs/FEASIBILITY.md`` in the repository for the analysis
and the repository ``README.md`` for installation and usage.

The package is importable without ComfyUI: :mod:`comfyui_cvrr.cvrr_core` is pure
PyTorch, and the node mappings are only produced when ComfyUI itself is
importable (which it always is when ComfyUI loads this directory as a custom
node pack).
"""

import importlib.util
import sys

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.1.0"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _comfyui_available() -> bool:
    """Is this package being imported from inside ComfyUI?"""
    try:
        return importlib.util.find_spec("folder_paths") is not None
    except (ImportError, ValueError):  # a stubbed/namespace module without __spec__
        return "folder_paths" in sys.modules


if _comfyui_available():
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F811
