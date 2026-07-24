"""Vendored, minimal subset of Depth-Anything-3 for the MetaView ComfyUI nodes.

Only the model definition + preprocessing/inference path is vendored (from
MetaView repo: DepthAnything3/src/depth_anything_3, ByteDance, Apache-2.0). The
export subpackage (glb/gs/npz/colmap/feat_vis) and the app/bench/services trees
were intentionally omitted because they pull in heavy deps (moviepy, etc.) not
present in the ComfyUI venv and are never used for conditioning.

The vendored package uses absolute ``depth_anything_3.*`` imports, so this
module makes its own directory importable as the top-level ``depth_anything_3``
package. Import ``get_depth_anything3`` from here rather than importing the
vendored package directly.
"""
import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_on_path():
    if _VENDOR_DIR not in sys.path:
        # Insert at front so the vendored copy wins over any other install.
        sys.path.insert(0, _VENDOR_DIR)


def get_depth_anything3():
    """Return the vendored ``DepthAnything3`` class (import on first use)."""
    _ensure_on_path()
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3
