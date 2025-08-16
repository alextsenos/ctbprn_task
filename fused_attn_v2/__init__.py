"""fused_attn_v2 package public interface.

Running this file directly is *not* recommended, but the fallback below lets it
work in a pinch by falling back to local imports when no parent package is
known.
"""

# Prefer relative imports when `fused_attn_v2` is imported as a package.
try:
    from .fused_v2 import FusedV2HRMModel  # type: ignore
    from .build_fused_v2 import FusedV2Config, build_fused_v2, pick_cuda_dtype  # type: ignore
except ImportError:  # pragma: no cover – executed directly / no parent pkg
    # Fallback: treat current directory as top-level on sys.path so absolute
    # imports resolve.
    import os as _os, sys as _sys
    _p = _os.path.dirname(__file__)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
    from fused_v2 import FusedV2HRMModel  # type: ignore
    from build_fused_v2 import FusedV2Config, build_fused_v2, pick_cuda_dtype  # type: ignore
__all__ = ["FusedV2HRMModel", "FusedV2Config", "build_fused_v2", "pick_cuda_dtype"]
