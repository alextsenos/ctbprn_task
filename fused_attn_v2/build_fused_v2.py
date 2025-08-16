from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Dual import shim: allow usage both as a package (preferred) and when this file
# is executed directly.
# ──────────────────────────────────────────────────────────────────────────────
try:
    # Normal package import
    from .fused_v2 import FusedV2HRMModel
except ImportError:  # pragma: no cover — fallback path for script execution
    import os, sys
    _p = os.path.dirname(__file__)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from fused_v2 import FusedV2HRMModel


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(s + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(torch.sigmoid(b) * F.silu(a)))

def pick_cuda_dtype() -> torch.dtype:
    """Prefer BF16 if supported else FP16. Uses PyTorch capability check."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing to run on CPU.")
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except AttributeError:
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16


# ──────────────────────────────────────────────────────────────────────────────
# Config (extends your original with HRM knobs, defaults sized for 16GB VRAM)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FusedV2Config:
    # fused attention (your original knobs)
    d_model: int = 2048
    n_heads: int = 32
    head_dim: int = 128
    d_value: int = 128
    r_latent: int = 64
    window: int = 8192
    block_size: int = 512
    top_k_blocks: int = 2
    tau_skip: float = 0.2

    # batch
    bsz: int = 1

    # feed-forward in fused blocks
    d_ff: int = 4 * 2048
    dropout: float = 0.0

    # HRM recurrence
    n_layers_L: int = 2          # fast/low-level layers (built from FusedAttentionV2)
    n_layers_H: int = 2          # slow/high-level layers
    N_cycles: int = 2            # H updates per segment
    T_steps: int = 2             # L steps per H update
    reset_L_each_cycle: bool = True

    # heads & ACT (halting)
    per_token_head: bool = False
    vocab_size: int = 0          # set >0 to enable a vocab head; else returns features
    use_act: bool = True
    Mmax: int = 4
    epsilon: float = 0.1

    # misc
    use_autocast: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Fused block = your FusedAttentionV2 + norms + SwiGLU FFN
# ──────────────────────────────────────────────────────────────────────────────

class FusedBlock(nn.Module):
    """
    Wraps your FusedAttentionV2 to behave like a transformer block:
      norm -> fused attention -> residual; norm -> ffn -> residual
    We keep the attention's internal state (e.g., latent/memory) outside in `state`.
    """
    def __init__(self, cfg: FusedV2Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = FusedAttentionV2(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            d_value=cfg.d_value,
            r_latent=cfg.r_latent,
            window=cfg.window,
            block_size=cfg.block_size,
            top_k_blocks=cfg.top_k_blocks,
            tau_skip=cfg.tau_skip,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn  = SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)

    def init_state(self, bsz: int, device: torch.device, dtype: torch.dtype):
        # delegate to your attention module
        return self.attn.init_state(bsz=bsz, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, state: Any, attn_mask: Optional[torch.Tensor] = None):
        """
        `state` is the per-block attention state created by init_state.
        """
        h = self.norm1(x)
        # be tolerant to different forward signatures of FusedAttentionV2
        try:
            h2, new_state = self.attn(h, state=state, attn_mask=attn_mask)
        except TypeError:
            # fallback: maybe forward(x, state) without mask
            h2, new_state = self.attn(h, state)
        x = x + self.drop(h2)

        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x, new_state


class FusedStack(nn.Module):
    def __init__(self, cfg: FusedV2Config, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([FusedBlock(cfg) for _ in range(n_layers)])

    def init_state(self, bsz: int, device: torch.device, dtype: torch.dtype) -> List[Any]:
        return [blk.init_state(bsz, device, dtype) for blk in self.blocks]

    def forward(self, x: torch.Tensor, states: List[Any], attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Any]]:
        new_states = []
        for blk, st in zip(self.blocks, states):
            x, st_new = blk(x, st, attn_mask=attn_mask)
            new_states.append(st_new)
        return x, new_states


# ──────────────────────────────────────────────────────────────────────────────
# HRM core built from the fused stacks
# ──────────────────────────────────────────────────────────────────────────────

class HRMCore(nn.Module):
    """
    Representation-level HRM (no token embedding): expects input features (B, L, D).
    Two recurrent stacks:
      - L_net (fast) runs T steps within each H cycle
      - H_net (slow) updates after each group of T L-steps
    One-step gradient approximation: only the last internal step is differentiable.
    """
    def __init__(self, cfg: FusedV2Config):
        super().__init__()
        self.cfg = cfg
        self.proj_x_to_L = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_H_to_L = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_L_to_H = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.L_net = FusedStack(cfg, n_layers=cfg.n_layers_L)
        self.H_net = FusedStack(cfg, n_layers=cfg.n_layers_H)

        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) if cfg.vocab_size > 0 else None
        self.q_head  = nn.Linear(cfg.d_model, 2, bias=True) if cfg.use_act else None

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / max(1, int(m.weight.size(-1)))); 
                if m.bias is not None: nn.init.zeros_(m.bias)

    def init_state(self, bsz: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        zH = torch.zeros(bsz, seq_len, self.cfg.d_model, device=device, dtype=dtype)
        zL = torch.zeros_like(zH)
        L_states = self.L_net.init_state(bsz, device, dtype)
        H_states = self.H_net.init_state(bsz, device, dtype)
        return {"zH": zH, "zL": zL, "L_states": L_states, "H_states": H_states}

    # ── one internal update of L and H, used by forward_segment ───────────────
    def _L_update(self, zL, zH, x, L_states, attn_mask):
        merged = zL + self.proj_H_to_L(zH) + self.proj_x_to_L(x)
        out, L_states = self.L_net(merged, L_states, attn_mask)
        return out, L_states

    def _H_update(self, zH, zL, H_states, attn_mask):
        merged = zH + self.proj_L_to_H(zL)
        out, H_states = self.H_net(merged, H_states, attn_mask)
        return out, H_states

    def _reset_L(self, zL):
        return torch.zeros_like(zL) if self.cfg.reset_L_each_cycle else zL

    # ── one "segment" with one-step gradient approximation ────────────────────
    def forward_segment(
        self,
        x_feats: torch.Tensor,                           # input features (B, L, D)
        state: Dict[str, Any],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg
        zH, zL = state["zH"], state["zL"]
        L_states, H_states = state["L_states"], state["H_states"]

        # all but the last internal step under no_grad
        with torch.no_grad():
            step_total = cfg.N_cycles * cfg.T_steps - 1
            for i in range(max(0, step_total)):
                zL, L_states = self._L_update(zL, zH, x_feats, L_states, attn_mask)
                if (i + 1) % cfg.T_steps == 0:
                    zH, H_states = self._H_update(zH, zL, H_states, attn_mask)
                    zL = self._reset_L(zL)

        # final step WITH grad
        zL, L_states = self._L_update(zL, zH, x_feats, L_states, attn_mask)
        zH, H_states = self._H_update(zH, zL, H_states, attn_mask)

        # outputs
        if self.cfg.per_token_head:
            logits = self.lm_head(self.norm_out(zH)) if self.lm_head is not None else self.norm_out(zH)
        else:
            pooled = self.norm_out(zH).mean(dim=1)
            logits = self.lm_head(pooled) if self.lm_head is not None else pooled

        q_values = self.q_head(self.norm_out(zH).mean(dim=1)) if self.q_head is not None else None

        new_state = {"zH": zH, "zL": zL, "L_states": L_states, "H_states": H_states}
        return new_state, logits, q_values


# ──────────────────────────────────────────────────────────────────────────────
# Public wrapper that matches the original builder’s return contract
# ──────────────────────────────────────────────────────────────────────────────

class FusedV2HRMModel(nn.Module):
    """
    HF-ish interface:
      - forward(x_feats, attention_mask=None, return_logits=True)
      - init_state(bsz, seq_len, device, dtype)
    This model operates on features (B, L, D). If you need token embeddings,
    embed outside and feed the hidden states here.
    """
    def __init__(self, cfg: FusedV2Config):
        super().__init__()
        self.cfg = cfg
        self.core = HRMCore(cfg)

    def init_state(self, bsz: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        return self.core.init_state(bsz, seq_len, device, dtype)

    def forward(
        self,
        x_feats: torch.Tensor,
        state: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
        return_q: bool = False,
    ) -> Tuple[Dict[str, Any], torch.Tensor, Optional[torch.Tensor]]:
        return self.core.forward_segment(x_feats, state, attention_mask if attention_mask is not None else None)


# ──────────────────────────────────────────────────────────────────────────────
# Builder (keeps your signature; returns model, state, device, dtype)
# ──────────────────────────────────────────────────────────────────────────────

def build_fused_v2(cfg: FusedV2Config, device_index: int = 0, dtype: Optional[torch.dtype] = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — refusing to run on CPU. [PROVED]")

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dtype = dtype or pick_cuda_dtype()

    # model sized for your GPU (bf16 on 5060 Ti 16GB if available)
    model = FusedV2HRMModel(cfg).to(device=device, dtype=dtype)

    # initialize HRM state (zH/zL + per-layer attention states)
    # sequence length is not known here; pick a sensible default and re-init at runtime
    # if your actual L differs.
    default_seq_len = cfg.block_size  # you can override when you have real inputs
    state = model.init_state(bsz=cfg.bsz, seq_len=default_seq_len, device=device, dtype=dtype)

    return model, state, device, dtype  # [PROVED]
