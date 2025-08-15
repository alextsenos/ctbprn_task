from dataclasses import dataclass
import torch

# Dual import shim: allow usage both as a package (preferred) and when this file is executed directly
try:
    # Normal package import
    from .fused_v2 import FusedAttentionV2
except ImportError:  # pragma: no cover — fallback path for script execution
    import os, sys
    _p = os.path.dirname(__file__)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from fused_v2 import FusedAttentionV2

@dataclass
class FusedV2Config:
    d_model: int = 2048
    n_heads: int = 32
    head_dim: int = 128
    d_value: int = 128
    r_latent: int = 64
    window: int = 8192
    block_size: int = 512
    top_k_blocks: int = 2
    tau_skip: float = 0.2
    bsz: int = 1

def pick_cuda_dtype() -> torch.dtype:
    """Prefer BF16 if supported else FP16. Uses PyTorch capability check."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing to run on CPU.")
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except AttributeError:
        # Fallback for older PyTorch: check major capability (Ampere is 8.0+)
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16

from typing import Optional

def build_fused_v2(cfg: FusedV2Config, device_index: int = 0, dtype: Optional[torch.dtype] = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing to run on CPU. [PROVED]")
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dtype = dtype or pick_cuda_dtype()

    attn = FusedAttentionV2(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        d_value=cfg.d_value,
        r_latent=cfg.r_latent,
        window=cfg.window,
        block_size=cfg.block_size,
        top_k_blocks=cfg.top_k_blocks,
        tau_skip=cfg.tau_skip,
    ).to(device=device, dtype=dtype)

    state = attn.init_state(bsz=cfg.bsz, device=device, dtype=dtype)
    return attn, state, device, dtype  # [PROVED]
