# fused_v2.py
# -----------------------------------------------------------------------------
# FusedV2 + HRM for long-context reasoning on mid-range GPUs (e.g., RTX 5060 Ti 16GB).
# Branches:
#   - Local block attention (sliding window)
#   - Dilated block attention
#   - Latent attention (tokens <-> latent slots)
#   - Infini-style memory (compress + retrieve from memory tokens)
#
# Mixer:
#   - "TitanFusion": a soft / (optionally top-k) gated mixture over branches
#
# HRM core:
#   - Two recurrent modules: L (fast) and H (slow), each an encoder-only stack built
#     from the fused layer above.
#   - One-step gradient approximation (all but the last internal step run under no_grad).
#   - Deep supervision across segments with hidden-state detach between segments.
#   - Optional ACT/Q-head to learn a halting policy.
#
# Notes:
# - This is a practical, faithful integration of HRM (one-step grad, deep supervision, ACT).
#   See the paper’s pseudocode and training details.  [citations in your PR/README]
# - The attention masks are implemented with torch additive masks; for production
#   switch to FlashAttention v3 or a custom Triton kernel for speed.
# - Defaults are chosen to fit ~16GB VRAM with bf16/fp16 at typical seq lengths;
#   tune d_model, heads, windows, and layers for your exact workload.
# -----------------------------------------------------------------------------


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- utilities --------------------

def autocast_dtype():
    """Pick the best mixed-precision dtype available on the current GPU."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., d)
        s = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(s + self.eps)
        return self.weight * x


def swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(swiglu(self.up(x))))


# -------------------- attention helpers --------------------

def sliding_window_mask(L: int, window: int, device: torch.device) -> torch.Tensor:
    """(L, L) boolean mask where True = masked/blocked"""
    i = torch.arange(L, device=device)
    j = i[None, :] - i[:, None]
    return j.abs() > window  # (L, L) bool; True = masked


def dilated_mask(L: int, window: int, dilation: int, device: torch.device) -> torch.Tensor:
    """
    (L, L) boolean mask for a 'dilated' band: keep positions within ±window
    but only every `dilation`th index from the center line.
    Returns: (L, L) bool where True = masked/blocked
    """
    i = torch.arange(L, device=device)
    j = i[None, :] - i[:, None]
    keep = (j.abs() <= window) & ((j % dilation) == 0)
    return ~keep  # True = masked


def key_padding_from_attention_mask(attn_mask_01: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert HuggingFace-style 1=keep, 0=pad mask (B, L) -> MultiheadAttention key_padding_mask (B, L) with True for PAD.
    """
    if attn_mask_01 is None:
        return None
    return (attn_mask_01 == 0)


class MHA(nn.Module):
    """Wrapper around MultiheadAttention with batch_first=True and post-norm residual."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, bias=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,         # (Lq, Lkv)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Lkv) True=PAD
    ) -> torch.Tensor:
        out, _ = self.mha(x_q, x_kv, x_kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(out)


# -------------------- latent attention (DeepSeek-style) --------------------

class LatentPool(nn.Module):
    """
    Small set of K latent slots that first **read** tokens (latents <- tokens) and then
    **write** back (tokens <- latents). Similar spirit to Perceiver latent bottlenecks.
    """
    def __init__(self, d_model: int, n_heads: int, K: int, dropout: float = 0.0):
        super().__init__()
        self.K = K
        self.latents = nn.Parameter(torch.randn(1, K, d_model) / math.sqrt(d_model))
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.to_latents = MHA(d_model, n_heads, dropout)
        self.from_latents = MHA(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        lat = self.latents.expand(B, self.K, D)

        # latents read tokens
        lat_q = self.norm_q(lat)
        tok_kv = self.norm_kv(x)
        lat = lat + self.to_latents(lat_q, tok_kv, attn_mask=None, key_padding_mask=key_padding_mask)

        # tokens read latents
        x = x + self.from_latents(self.norm_q(x), self.norm_kv(lat), attn_mask=None, key_padding_mask=None)
        return x, lat


# -------------------- infini-style memory --------------------

class InfiniMemory(nn.Module):
    """
    Lightweight memory compressor:
      - Builds M memory tokens via attention pooling from tokens.
      - Updates memory with EMA (gamma) to carry signal across layers/segments.
    """
    def __init__(self, d_model: int, n_heads: int, M: int, dropout: float = 0.0, gamma: float = 0.9):
        super().__init__()
        self.M = M
        self.gamma = gamma
        self.mem_tokens = nn.Parameter(torch.randn(1, M, d_model) / math.sqrt(d_model))
        self.read_mha = MHA(d_model, n_heads, dropout)
        self.write_mha = MHA(d_model, n_heads, dropout)
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor], carry: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        mem_prev = self.mem_tokens.expand(B, self.M, D) if carry is None else carry

        # Write: mem reads tokens
        mem_new = mem_prev + self.write_mha(self.norm_q(mem_prev), self.norm_kv(x), key_padding_mask=key_padding_mask)
        mem = self.gamma * mem_prev + (1.0 - self.gamma) * mem_new

        # Read: tokens read memory
        x = x + self.read_mha(self.norm_q(x), self.norm_kv(mem), key_padding_mask=None)
        return x, mem


# -------------------- moBA branches + TitanFusion --------------------

class LocalDilatedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, window: int = 256, dilation: int = 4):
        super().__init__()
        self.local = MHA(d_model, n_heads, dropout)
        self.dilated = MHA(d_model, n_heads, dropout)
        self.window = window
        self.dilation = dilation
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        device = x.device

        # build additive masks once per forward
        local_mask = sliding_window_mask(L, self.window, device)
        dil_mask = dilated_mask(L, self.window, self.dilation, device)

        q = self.norm_q(x)
        k = self.norm_kv(x)
        out_local = self.local(q, k, attn_mask=local_mask, key_padding_mask=key_padding_mask)
        out_dil = self.dilated(q, k, attn_mask=dil_mask, key_padding_mask=key_padding_mask)
        return out_local, out_dil


class TitanFusion(nn.Module):
    """
    Gated mixture over branches: local, dilated, latent, memory.
    Optionally enforces top-k sparsity over branches (per token).
    """
    def __init__(self, d_model: int, n_branches: int, dropout: float = 0.0, topk: int = 0):
        super().__init__()
        self.gate = nn.Linear(d_model, n_branches, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.topk = topk

    def forward(self, x: torch.Tensor, branches: List[torch.Tensor]) -> torch.Tensor:
        # branches: list of (B, L, D)
        B, L, D = x.shape
        H = torch.stack(branches, dim=-2)  # (B, L, n_br, D)
        logits = self.gate(x)              # (B, L, n_br)
        if self.topk and self.topk < logits.size(-1):
            # keep top-k branches per token
            topk_vals, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            logits = mask
        w = torch.softmax(logits, dim=-1).unsqueeze(-1)  # (B, L, n_br, 1)
        fused = (w * H).sum(dim=-2)  # (B, L, D)
        return self.dropout(fused)


# -------------------- fused layer (one block) --------------------

class FusedLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        window: int,
        dilation: int,
        K_latent: int,
        M_mem: int,
        titan_topk: int = 0,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)

        self.local_dil = LocalDilatedAttention(d_model, n_heads, dropout, window, dilation)
        self.latent = LatentPool(d_model, n_heads, K_latent, dropout)
        self.mem = InfiniMemory(d_model, n_heads, M_mem, dropout)

        self.mixer = TitanFusion(d_model, n_branches=4, dropout=dropout, topk=titan_topk)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        mem_carry: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          x_out, latents, mem_new
        """
        h = self.norm(x)

        # moBA branches
        out_local, out_dil = self.local_dil(h, key_padding_mask)
        x_lat, latents = self.latent(h, key_padding_mask)
        x_mem, mem_new = self.mem(h, key_padding_mask, carry=mem_carry)

        fused = self.mixer(h, [out_local, out_dil, x_lat, x_mem])
        x = x + self.dropout(fused)
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x, latents, mem_new


class FusedStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        window: int,
        dilation: int,
        K_latent: int,
        M_mem: int,
        titan_topk: int,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FusedLayer(d_model, n_heads, d_ff, dropout, window, dilation, K_latent, M_mem, titan_topk)
            for _ in range(n_layers)
        ])
        self.grad_ckpt = grad_ckpt

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        mem_carry: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.contiguous()  # Ensure input is contiguous
        
        def _run_blk(blk, _x, _mask, _m):
            # Pass all inputs explicitly, no outer scope captures
            out_x, lat_last, new_m = blk(_x, key_padding_mask=_mask, mem_carry=_m)
            return out_x, lat_last, new_m

        lat_last = None
        for blk in self.layers:
            if self.grad_ckpt and x.requires_grad:
                # Create empty tensors for None inputs to maintain consistent signature
                mask_tensor = key_padding_mask if key_padding_mask is not None else torch.tensor(
                    [], device=x.device, dtype=torch.bool
                )
                mem_tensor = mem_carry if mem_carry is not None else torch.tensor(
                    [], device=x.device, dtype=x.dtype
                )
                
                # Debug prints (can be removed after verification)
                # print("x", tuple(x.shape), x.dtype, x.device)
                # if key_padding_mask is not None:
                #     print("kpm", tuple(key_padding_mask.shape), key_padding_mask.dtype)
                # if mem_carry is not None and torch.is_tensor(mem_carry):
                #     print("mem", tuple(mem_carry.shape), mem_carry.dtype)
                
                x, lat_last, mem_carry = torch.utils.checkpoint.checkpoint(
                    _run_blk,
                    blk,  # Pass block as first argument
                    x,    # Input tensor
                    mask_tensor,  # Mask tensor (or empty)
                    mem_tensor,   # Memory tensor (or empty)
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                
                # Restore None for empty tensors
                if mem_carry.numel() == 0:
                    mem_carry = None
            else:
                x, lat_last, mem_carry = blk(x, key_padding_mask, mem_carry)
                
        return x, mem_carry


# -------------------- HRM Core --------------------

@dataclass
class HRMConfig:
    # embedding
    vocab_size: int = 32000
    max_len: int = 8192
    pad_token_id: int = 0

    # model dims
    d_model: int = 640
    n_heads: int = 8
    d_ff: int = 2048

    # L/H stacks
    n_layers_L: int = 6
    n_layers_H: int = 4

    # fused attention params
    dropout: float = 0.0
    window: int = 256
    dilation: int = 4
    K_latent: int = 64
    M_mem: int = 64
    titan_topk: int = 0
    grad_ckpt: bool = True

    # recurrence
    N_cycles: int = 2          # H-updates
    T_steps: int = 2           # L-steps per H
    reset_L_each_cycle: bool = True

    # heads
    per_token_head: bool = False

    # ACT (halting)
    use_act: bool = True
    Mmax: int = 4
    epsilon: float = 0.1

    # autocast + device
    use_autocast: bool = True


class HRM(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        # merge projections
        self.proj_x_to_L = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_H_to_L = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_L_to_H = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # stacks
        self.L_net = FusedStack(
            cfg.d_model, cfg.n_layers_L, cfg.n_heads, cfg.d_ff, cfg.dropout,
            cfg.window, cfg.dilation, cfg.K_latent, cfg.M_mem, cfg.titan_topk, cfg.grad_ckpt
        )
        self.H_net = FusedStack(
            cfg.d_model, cfg.n_layers_H, cfg.n_heads, cfg.d_ff, cfg.dropout,
            cfg.window, cfg.dilation, cfg.K_latent, cfg.M_mem, cfg.titan_topk, cfg.grad_ckpt
        )

        # output head
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ACT Q-head
        self.q_head = nn.Linear(cfg.d_model, 2, bias=True) if cfg.use_act else None

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.size(-1)))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.embedding_dim))

    # ----- helpers -----
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        return self.tok_emb(input_ids) + self.pos_emb(pos)

    def _L_update(self, zL: torch.Tensor, zH: torch.Tensor, x_emb: torch.Tensor,
                  key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        merged = zL + self.proj_H_to_L(zH) + self.proj_x_to_L(x_emb)
        out, mem = self.L_net(merged, key_padding_mask, mem_carry=None)
        return out, mem

    def _H_update(self, zH: torch.Tensor, zL: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        merged = zH + self.proj_L_to_H(zL)
        out, mem = self.H_net(merged, key_padding_mask, mem_carry=None)
        return out, mem

    def _reset_L(self, zL: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(zL) if self.cfg.reset_L_each_cycle else zL

    # ----- segment forward with one-step gradient -----
    def forward_segment(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Runs one "segment" of HRM.
        Returns:
          new_state (zH, zL), logits, q_values (halt/continue) or None
        """
        cfg = self.cfg
        zH, zL = state
        x_emb = self.embed(input_ids)

        kpm = key_padding_from_attention_mask(attention_mask)

        # All but last step under no_grad (1-step gradient approximation)
        with torch.no_grad():
            for i in range(cfg.N_cycles * cfg.T_steps - 1):
                zL, _ = self._L_update(zL, zH, x_emb, kpm)
                if (i + 1) % cfg.T_steps == 0:
                    zH, _ = self._H_update(zH, zL, kpm)
                    zL = self._reset_L(zL)

        # Final step WITH grad
        zL, _ = self._L_update(zL, zH, x_emb, kpm)
        zH, _ = self._H_update(zH, zL, kpm)

        if cfg.per_token_head:
            logits = self.lm_head(self.norm_out(zH))  # (B, L, V)
        else:
            pooled = self.norm_out(zH).mean(dim=1)    # (B, D)
            logits = self.lm_head(pooled)             # (B, V)

        q_values = None
        if self.q_head is not None:
            pooled = self.norm_out(zH).mean(dim=1)
            q_values = self.q_head(pooled)  # (B, 2)
        return (zH, zL), logits, q_values

    def init_state(self, B: int, L: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        return torch.zeros(B, L, self.cfg.d_model, device=dev), torch.zeros(B, L, self.cfg.d_model, device=dev)


# -------------------- training helpers --------------------

def sequence_loss(logits: torch.Tensor, targets: torch.Tensor, per_token_head: bool) -> torch.Tensor:
    if not per_token_head:
        return F.cross_entropy(logits, targets)
    B, L, V = logits.shape
    return F.cross_entropy(logits.reshape(B * L, V), targets.reshape(B * L))


@torch.no_grad()
def _epsilon_min_segments(epsilon: float, Mmax: int) -> int:
    if torch.rand(()) < epsilon and Mmax >= 2:
        return int(torch.randint(2, Mmax + 1, (1,)).item())
    return 1


def deep_supervision_step(
    model: HRM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    state: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    use_act: bool,
    Mmax: int,
    epsilon: float,
    *,
    step: int = 0,
    act_warmup_steps: int = 200,
    min_segments: int = 2,
    q_weight: float = 0.25,
    halt_margin: float = 0.1,
) -> Dict[str, Any]:
    """
    One training iteration across one or more supervision segments with ACT warmup.
    
    Args:
        model: The HRM model
        input_ids: Input token IDs [B, L]
        targets: Target labels [B] or [B, L] if per_token_head
        attention_mask: Attention mask [B, L]
        state: Current (zH, zL) state tuple
        optimizer: Optimizer for the model
        use_act: Whether to use ACT
        Mmax: Maximum number of segments
        epsilon: Epsilon for minimum segments sampling
        step: Current training step (for warmup)
        act_warmup_steps: Steps before enabling ACT
        min_segments: Minimum segments to run
        q_weight: Weight for Q-loss
        halt_margin: Margin required for halting
    """
    cfg = model.cfg
    device = input_ids.device

    # ACT warmup: disable ACT entirely for first act_warmup_steps
    act_enabled = use_act and (step >= act_warmup_steps)

    # epsilon sampling for Mmin when ACT is enabled, else fixed
    def _sample_Mmin():
        if not act_enabled:
            return max(1, min_segments)
        base = _epsilon_min_segments(epsilon, Mmax)
        return max(base, min_segments)

    m = 0
    halted = False
    total_seq = 0.0
    total_q = 0.0
    last_logits = None
    last_q = None

    Mmin = _sample_Mmin()
    while True:
        m += 1
        state, logits, q_values = model.forward_segment(state, input_ids, attention_mask)

        # ---- sequence loss (FP32 for stability)
        if not cfg.per_token_head:
            # logits: (B, V) ; targets: (B,)
            assert logits.dim() == 2 and logits.size(0) == input_ids.size(0), f"logits {tuple(logits.shape)}"
            seq_loss = F.cross_entropy(logits.float(), targets.long())
        else:
            # logits: (B, L, V) ; targets: (B, L)
            B, L, V = logits.shape
            assert targets.shape == (B, L), f"targets {tuple(targets.shape)} != {(B, L)}"
            seq_loss = F.cross_entropy(logits.reshape(B * L, V).float(),
                                     targets.reshape(B * L).long())

        # ---- Q-loss with logits (only if ACT enabled)
        q_loss = torch.tensor(0.0, device=device)
        do_halt = (m >= Mmax)
        if act_enabled and (q_values is not None):
            halt_logit, cont_logit = q_values[:, 0], q_values[:, 1]

            with torch.no_grad():
                if not cfg.per_token_head:
                    reward = (logits.argmax(dim=-1) == targets).float()  # (B,)
                else:
                    pred_tok = logits.argmax(dim=-1)
                    reward = (pred_tok.eq(targets).all(dim=1)).float()
                target_halt = reward
                target_continue = torch.zeros_like(reward)

            q_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(halt_logit, target_halt) +
                F.binary_cross_entropy_with_logits(cont_logit, target_continue)
            ) * q_weight

            # require a margin for halting, and obey Mmin / Mmax
            want_halt = (m >= Mmin) & (halt_logit > (cont_logit + halt_margin))
            must_halt = torch.full_like(want_halt, m >= Mmax, dtype=torch.bool)
            do_halt = (want_halt | must_halt).all().item()

        # ---- optimize
        loss = (seq_loss + q_loss).float()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_seq += seq_loss.detach().float().item()
        total_q   += q_loss.detach().float().item()
        last_logits = logits.detach()
        last_q = q_values.detach() if q_values is not None else None

        # detach state between segments
        state = (state[0].detach(), state[1].detach())

        if do_halt:
            halted = True
            break

    return {
        "halted": halted,
        "segments": m,
        "seq_loss": total_seq / max(1, m),
        "q_loss": total_q / max(1, m),
        "last_logits": last_logits,
        "last_q": last_q,
        "state": state,
    }


# -------------------- full model wrapper --------------------

class FusedV2HRMModel(nn.Module):
    """
    Convenience wrapper that exposes a HF-like API:
      forward(input_ids, attention_mask=None, labels=None, return_loss=False, **kw)
    """
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.core = HRM(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, Any]:
        B, L = input_ids.shape
        state = self.core.init_state(B, L, device=input_ids.device)

        # One segment forward (for eval/inference). Training should use deep_supervision_step().
        state, logits, q_values = self.core.forward_segment(state, input_ids, attention_mask)

        out = {"logits": logits, "q_values": q_values}
        if return_loss and labels is not None:
            out["loss"] = sequence_loss(logits, labels, self.cfg.per_token_head)
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Greedy or temperature sampling over sequence-level head.
        If per_token_head=True, this method does per-step token sampling.
        """
        cfg = self.cfg
        B, L = input_ids.shape
        device = input_ids.device

        if not cfg.per_token_head:
            # sequence-level: run segments and argmax a single label
            state = self.core.init_state(B, L, device=device)
            state, logits, _ = self.core.forward_segment(state, input_ids, attention_mask)
            return logits.argmax(dim=-1)

        # token-level generation (minimal example)
        seq = input_ids.clone()
        for _ in range(max_new_tokens):
            Lcur = seq.size(1)
            state = self.core.init_state(B, Lcur, device=device)
            _, logits, _ = self.core.forward_segment(state, seq, attention_mask=None)
            next_logits = logits[:, -1]
            if temperature > 0:
                probs = torch.softmax(next_logits / max(1e-5, temperature), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
        return seq
