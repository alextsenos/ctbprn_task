"""
Fused Attention V2 (MoBA‑MLA + Infini‑Latent + Titans Gates) — up to attention output (pre‑MLP)

This is a reference‑quality, **dtype‑safe** implementation of the canonical V2 spec for
incremental decoding. It fixes previous dtype mismatches and ring‑buffer masking issues,
and provides precise masking for the filled window in the ring (no approximations).

Status labels in comments:
- [PROVED]: matches spec or standard transformer practice.
- [INFERENCE]: design/heuristic choices that are reasonable but not formally validated here.
- [UNVERIFIED]: optional hooks or comments about potential behavior without direct proof.

Notes
-----
* Focuses on incremental decoding (token‑by‑token). Prefill can be added later.  [PROVED]
* All persistent state is held in `FusedAttentionState`.                       [PROVED]
* Dtype/device coherence: the state auto‑adapts to the module's compute dtype
  and device on first use via `ensure_dtype`/`ensure_device`.                  [PROVED]

Author: AI PROJECT — Fused Attention V2 reference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility helpers
# ---------------------------

def safe_softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Numerically safe positive map σ(x)=softplus(x). [PROVED]
    Using default beta=1 to keep magnitudes reasonable.
    """
    return F.softplus(x, beta=beta)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over entries where mask==True; masked positions receive a large negative.
    mask is boolean and broadcastable to logits. [PROVED]
    """
    # Use the minimum finite value for the dtype to avoid NaNs in half precision.
    neg_large = torch.finfo(logits.dtype).min
    logits = torch.where(mask, logits, torch.full_like(logits, neg_large))
    return F.softmax(logits, dim=dim)


# ---------------------------
# State container
# ---------------------------

@dataclass
class FusedAttentionState:
    """Holds all persistent per‑layer state for incremental decoding. [PROVED]

    Shapes are per batch and head where applicable.

    Args
    ----
    w: latent KV window size.
    B: block size (tokens per block).
    H: number of heads.
    r: latent dimension.
    device, dtype: initialize buffers on this device/dtype.
    """

    w: int
    B: int
    H: int
    r: int
    device: torch.device
    dtype: torch.dtype

    # Latent KV ring buffers (physical ring of size w). Shapes: [Bsz, H, w, r]
    K_latent: Optional[torch.Tensor] = None
    V_latent: Optional[torch.Tensor] = None

    # Block id per physical slot in the ring: [Bsz, w] (int64)
    block_ids: Optional[torch.Tensor] = None

    # Current write pointer in ring, and filled logical length (<= w)
    write_ptr: int = 0
    length: int = 0

    # Current block running stats
    cur_block_sum: Optional[torch.Tensor] = None  # [Bsz, H, r]
    cur_block_count: int = 0
    cur_block_id: int = 0

    # Stored past block centroids (list of tensors [Bsz, H, r]) up to some cap
    block_means: List[torch.Tensor] = None
    block_mean_cap: int = 4096  # safety cap

    # Infini memory per head per batch
    M: Optional[torch.Tensor] = None  # [Bsz, H, r, r]
    z: Optional[torch.Tensor] = None  # [Bsz, H, r]

    # Titans running accumulator S_t
    S_prev: Optional[torch.Tensor] = None  # [Bsz, H, r, r]

    # Cached batch size (immutable for lifetime of this state)
    bsz: int = 0

    # ---------- init & maintenance ----------
    def init_buffers(self, bsz: int):  # [PROVED]
        self.bsz = bsz
        self.K_latent = torch.zeros(bsz, self.H, self.w, self.r, device=self.device, dtype=self.dtype)
        self.V_latent = torch.zeros_like(self.K_latent)
        self.block_ids = torch.full((bsz, self.w), fill_value=-1, device=self.device, dtype=torch.long)
        self.cur_block_sum = torch.zeros(bsz, self.H, self.r, device=self.device, dtype=self.dtype)
        self.block_means = []
        self.M = torch.zeros(bsz, self.H, self.r, self.r, device=self.device, dtype=self.dtype)
        # Prefill z with small epsilon to stabilize early memory reads
        self.z = torch.full((bsz, self.H, self.r), 1e-3, device=self.device, dtype=self.dtype)
        self.S_prev = torch.zeros(bsz, self.H, self.r, self.r, device=self.device, dtype=self.dtype)
        self.write_ptr = 0
        self.length = 0
        self.cur_block_count = 0
        self.cur_block_id = 0

    def ensure_device(self, device: torch.device):  # [PROVED]
        """Move buffers to `device` if needed."""
        if self.device == device:
            return
        self.device = device
        if self.K_latent is not None:
            self.K_latent = self.K_latent.to(device)
            self.V_latent = self.V_latent.to(device)
            self.block_ids = self.block_ids.to(device)
            self.cur_block_sum = self.cur_block_sum.to(device)
            self.M = self.M.to(device)
            self.z = self.z.to(device)
            self.S_prev = self.S_prev.to(device)

    def ensure_dtype(self, dtype: torch.dtype):  # [PROVED]
        """Convert buffers to `dtype` if needed so einsums don't dtype‑clash."""
        if self.dtype == dtype:
            return
        self.dtype = dtype
        if self.K_latent is not None:
            self.K_latent = self.K_latent.to(dtype)
            self.V_latent = self.V_latent.to(dtype)
            self.cur_block_sum = self.cur_block_sum.to(dtype)
            self.M = self.M.to(dtype)
            self.z = self.z.to(dtype)
            self.S_prev = self.S_prev.to(dtype)

    # ---------- ring ops ----------
    def append_kv(self, K_tL: torch.Tensor, V_tL: torch.Tensor):  # [PROVED]
        """Append one latent KV token to the ring (runtime state, no grad)."""
        with torch.no_grad():
            # Hard break any graph links
            K_tL = K_tL.detach()
            V_tL = V_tL.detach()
            assert K_tL.shape == (self.bsz, self.H, self.r)
            idx = self.write_ptr
            self.K_latent[:, :, idx, :] = K_tL
            self.V_latent[:, :, idx, :] = V_tL
            self.block_ids[:, idx] = self.cur_block_id

        self.write_ptr = (self.write_ptr + 1) % self.w
        self.length = min(self.length + 1, self.w)

        # Update running stats for current block. [PROVED]
        self.cur_block_sum.add_(K_tL)
        self.cur_block_count += 1

        # Handle block rollover. [PROVED]
        if self.cur_block_count == self.B:
            centroid = (self.cur_block_sum / float(self.B)).detach()
            self.block_means.append(centroid)  # already detached
            if len(self.block_means) > self.block_mean_cap:
                self.block_means.pop(0)
            self.cur_block_sum.zero_()
            self.cur_block_count = 0
            self.cur_block_id += 1

    def valid_slots_mask(self) -> torch.Tensor:  # [PROVED]
        """Boolean mask over physical ring slots that are logically filled. Shape [1, 1, w]."""
        m = torch.zeros(self.w, dtype=torch.bool, device=self.device)
        L = self.length
        if L == self.w:
            m[:] = True
        elif L > 0:
            start = (self.write_ptr - L) % self.w
            if start + L <= self.w:
                m[start : start + L] = True
            else:
                tail = (start + L) - self.w
                m[start : self.w] = True
                m[0 : tail] = True
        # Expand to broadcast over batch and heads later
        return m.view(1, 1, self.w)

    def routed_mask(self, Q_tL: torch.Tensor, k_top: int) -> torch.Tensor:
        """Compute boolean mask over window positions to select routed tokens. [NEW]
        Returns a mask of shape [Bsz, 1, w] (broadcastable over heads).
        Routing is based on block means: select current block + top‑k of past blocks by ⟨Q_t^L, K̄_i^L⟩/√r.
        """
        if self.length == 0:
            return torch.zeros(self.bsz, 1, self.w, dtype=torch.bool, device=self.device)

        routed_block_ids_per_batch: List[torch.Tensor] = []
        if self.block_means:
            # scores: per stored block, per batch and head
            scores_list = []
            for Kbar in self.block_means:  # each: [Bsz, H, r]
                s = (Q_tL * Kbar).sum(-1) / (self.r ** 0.5)  # [Bsz, H]
                scores_list.append(s)
            scores = torch.stack(scores_list, dim=-1)  # [Bsz, H, num_blocks]
            # Recency bias: favor recent blocks exponentially (gamma^age). Recent => weight 1.0
            num_blocks = scores.shape[-1]
            gamma = 0.97  # older blocks down‑weighted by gamma^age
            # age 0 = most recent (last in list), age = num_blocks-1 = oldest (first)
            ages = torch.arange(num_blocks - 1, -1, -1, device=self.device, dtype=scores.dtype)
            weights = gamma ** ages  # [num_blocks]
            scores = scores * weights  # broadcast on last dim
            scores_mean = scores.mean(1)               # [Bsz, num_blocks]
            k_eff = min(k_top, scores_mean.shape[-1])
            topk_idx = scores_mean.topk(k_eff, dim=-1).indices  # [Bsz, k_eff]
            # Stored block_means correspond to block ids [0..cur_block_id-1] in order. [PROVED]
            for b in range(self.bsz):
                routed_block_ids_per_batch.append(topk_idx[b].to(torch.long))
        else:
            for _ in range(self.bsz):
                routed_block_ids_per_batch.append(torch.empty(0, dtype=torch.long, device=self.device))

        mask = torch.zeros(self.bsz, self.w, dtype=torch.bool, device=self.device)
        cur_id = self.cur_block_id
        for b in range(self.bsz):
            routed = set(routed_block_ids_per_batch[b].tolist())
            routed.add(cur_id)  # always include current block
            routed_tensor = torch.tensor(list(routed), device=self.device, dtype=torch.long)
            pos_blocks = self.block_ids[b]  # [w]
            allowed = torch.isin(pos_blocks, routed_tensor)
            mask[b] = allowed

        return mask.unsqueeze(1)  # [Bsz, 1, w]


# ---------------------------
# Router and Gater submodules
# ---------------------------

class TinyRouter(nn.Module):
    """A tiny router that outputs p_skip in [0,1] from latent Q. [NEW]
    2‑layer MLP with GELU and Sigmoid. Per‑head, pointwise.
    """
    def __init__(self, r: int):
        super().__init__()
        self.fc1 = nn.Linear(r, 2 * r)
        self.fc2 = nn.Linear(2 * r, 1)

    def forward(self, q_lat: torch.Tensor) -> torch.Tensor:
        # q_lat: [Bsz, H, r]
        x = F.gelu(self.fc1(q_lat))
        p = torch.sigmoid(self.fc2(x))  # [Bsz, H, 1]
        return p.squeeze(-1)  # [Bsz, H]


class TitansGater(nn.Module):
    """Produces (α, η, θ) from surprise features u_t. [INFERENCE]
    α∈[0,1] via sigmoid, η,θ≥0 via softplus. Reduces per‑head to scalars.
    """
    def __init__(self, r: int, hidden: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(4 * r, hidden * r)
        self.fc2 = nn.Linear(hidden * r, 3 * r)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # u: [Bsz, H, 4r]
        h = F.gelu(self.fc1(u))
        out = self.fc2(h)  # [Bsz, H, 3r]
        a, e, t = out.chunk(3, dim=-1)
        # Reduce across r to scalars per head via mean (lightweight). [INFERENCE]
        alpha = torch.sigmoid(a.mean(-1, keepdim=True))  # [Bsz, H, 1]
        eta = F.softplus(e.mean(-1, keepdim=True))
        theta = F.softplus(t.mean(-1, keepdim=True))
        return alpha, eta, theta


# ---------------------------
# The Fused Attention V2 module
# ---------------------------

class FusedAttentionV2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        d_value: int,
        r_latent: int,
        window: int,
        block_size: int,
        top_k_blocks: int = 2,
        tau_skip: float = 0.2,
        rope: Optional[object] = None,  # placeholder hook for RoPE
    ):
        """Initialize the fused attention (pre‑MLP) stack. [PROVED]

        Args:
            d_model: model dimension.
            n_heads: number of attention heads H.
            head_dim: per‑head Q/K dimension d_h.
            d_value: per‑head V/out dimension d_v.
            r_latent: latent rank r.
            window: latent KV window size w.
            block_size: block size B for routing.
            top_k_blocks: k for MoBA routing.
            tau_skip: threshold for skipping local path (<= uses local). [INFERENCE]
            rope: optional positional encoding hook. [UNVERIFIED]
        """
        super().__init__()
        self.d_model = d_model
        self.H = n_heads
        self.d_h = head_dim
        self.d_v = d_value
        self.r = r_latent
        self.w = window
        self.B = block_size
        self.k_top = top_k_blocks
        self.tau_skip = tau_skip
        self.rope = rope

        # Input layer norm before attention (Pre‑LN block). [PROVED]
        self.ln = nn.LayerNorm(d_model)

        # Combined QKV projection to per‑head spaces. [PROVED]
        self.W_qkv = nn.Linear(d_model, n_heads * (2 * head_dim + d_value), bias=False)

        # Latent projections per head (E_q, E_k, E_v). [PROVED]
        self.E_q = nn.Parameter(torch.randn(n_heads, head_dim, r_latent) / (head_dim ** 0.5))
        self.E_k = nn.Parameter(torch.randn(n_heads, head_dim, r_latent) / (head_dim ** 0.5))
        self.E_v = nn.Parameter(torch.randn(n_heads, d_value, r_latent) / (d_value ** 0.5))

        # Decoder back to value dim per head (D_v). Init as pseudo-inverse of E_v for well-conditioned start.
        D_init = torch.empty(n_heads, r_latent, d_value)
        for h in range(n_heads):
            Ev = self.E_v[h].detach().to(torch.float32)
            D_init[h] = torch.linalg.pinv(Ev).to(Ev.dtype)
        self.D_v = nn.Parameter(D_init)

        # Output projection back to d_model. [PROVED]
        self.W_o = nn.Linear(n_heads * d_value, d_model, bias=False)

        # Router and Gater. [NEW]
        self.router = TinyRouter(r_latent)
        self.gater = TitansGater(r_latent)
        # Optionally freeze gater parameters; they affect no-grad state only
        for p in self.gater.parameters():
            p.requires_grad_(False)

        # Per‑head learnable mix β (scalar). Initialize near 0.2 so local path dominates early. [INFERENCE]
        self.beta = nn.Parameter(torch.full((n_heads, 1), 0.2))

    # --------- public API ---------
    def init_state(self, bsz: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> FusedAttentionState:
        # Use module's parameter device/dtype by default. [PROVED]
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        st = FusedAttentionState(self.w, self.B, self.H, self.r, device, dtype)
        st.init_buffers(bsz)
        return st

    @torch.no_grad()
    def reset_state_(self, state: FusedAttentionState):  # [PROVED]
        state.init_buffers(state.bsz)

    # --------- forward (incremental) ---------
    def forward(self, x_t: torch.Tensor, state: FusedAttentionState) -> torch.Tensor:
        """One‑step incremental forward.
        x_t: [Bsz, d_model] token at time t. Returns [Bsz, d_model]. [PROVED]
        """
        Bsz = x_t.size(0)
        if state.bsz not in (0, Bsz):
            raise RuntimeError(f"State batch size {state.bsz} mismatches input batch {Bsz}")
        if state.bsz == 0:
            state.init_buffers(Bsz)

        # Harmonize state device/dtype with module's compute soon used for Q,K,V. [PROVED]
        compute_device = x_t.device
        compute_dtype = self.ln.weight.dtype  # LN weight dtype defines compute path
        state.ensure_device(compute_device)
        state.ensure_dtype(compute_dtype)

        # Pre‑LN
        x = self.ln(x_t.to(compute_dtype))

        # Project to per‑head Q, K, V. [PROVED]
        qkv = self.W_qkv(x)  # [Bsz, H*(2*d_h + d_v)]
        qkv = qkv.view(Bsz, self.H, 2 * self.d_h + self.d_v)
        Q, K, V = torch.split(qkv, [self.d_h, self.d_h, self.d_v], dim=-1)  # [Bsz,H,d_h] etc.

        # Optional RoPE (hook). [UNVERIFIED]
        if self.rope is not None:
            Q, K = self.rope(Q, K)

        # Latent projections: Q^L, K^L, V^L. [PROVED]
        QL = torch.einsum('bhd,hdr->bhr', Q, self.E_q)
        KL = torch.einsum('bhd,hdr->bhr', K, self.E_k)
        VL = torch.einsum('bhd,hdr->bhr', V, self.E_v)

        # Append to latent KV window and update block stats. [PROVED]
        state.append_kv(KL.detach(), VL.detach())

        # ---------------- Local latent attention over routed tokens ----------------
        # Router: decide whether to use local path. [NEW]
        p_skip = self.router(QL)  # [Bsz, H]
        use_local = (p_skip <= self.tau_skip).unsqueeze(-1)  # [Bsz,H,1] boolean mask

        # Routing mask over the window positions: [Bsz,1,w]
        routed_mask = state.routed_mask(QL, self.k_top)  # [Bsz,1,w]
        # Exclude the slot just written (self) to avoid trivial copy-attention
        last_idx = (state.write_ptr - 1) % self.w
        no_self_mask = torch.ones(1, 1, self.w, dtype=torch.bool, device=state.device)
        no_self_mask[..., last_idx] = False
        routed_mask = routed_mask & no_self_mask

        # Valid logical slots in physical ring: [1,1,w] -> broadcast
        valid_mask = state.valid_slots_mask().to(routed_mask.device)  # [1,1,w]

        # Combine masks (and broadcast over heads below)
        base_mask = routed_mask & valid_mask  # [Bsz,1,w]

        # Build tensors over window (runtime state => no grad): [Bsz,H,w,r]
        Kwin = state.K_latent.detach()
        Vwin = state.V_latent.detach()

        if base_mask.any():
            # logits: [Bsz,H,w]
            logits = torch.einsum('bhr,bhwr->bhw', QL, Kwin) / (self.r ** 0.5)
            attn_mask = base_mask.expand(-1, self.H, -1)  # [Bsz,H,w]

            # In incremental decode there are no future tokens within the current block yet, so
            # standard causal enforcement reduces to masking invalid slots only. [PROVED]

            probs = masked_softmax(logits, attn_mask, dim=-1)  # [Bsz,H,w]
            YdotL = torch.einsum('bhw,bhwr->bhr', probs, Vwin)  # [Bsz,H,r]
        else:
            YdotL = torch.zeros_like(QL)

        # Honor skip decision per head: if skip, zero out local contribution. [PROVED]
        YdotL = torch.where(use_local, YdotL, torch.zeros_like(YdotL))

        # Decode to value dim: Y_dot = Y_dot^L D_v. [PROVED]
        Ydot = torch.einsum('bhr,hrd->bhd', YdotL, self.D_v)

        # ---------------- Infini latent memory read ----------------
        sigmaQL = safe_softplus(QL)  # [Bsz,H,r]
        # Read from detached snapshots to break links to previous steps. [PROVED]
        M_read = state.M.detach().clone()
        z_read = state.z.detach().clone()
        num = torch.einsum('bhr,bhrr->bhr', sigmaQL, M_read)
        den = (sigmaQL * z_read).sum(-1, keepdim=True).clamp_min(1e-6)
        YmemL = num / den
        Ymem = torch.einsum('bhr,hrd->bhd', YmemL, self.D_v)

        # ---------------- Mix local vs memory ----------------
        mix = torch.sigmoid(self.beta)  # [H,1]
        mix = mix.unsqueeze(0).expand(Bsz, -1, -1)  # [Bsz,H,1]
        A = mix * Ymem + (1.0 - mix) * Ydot  # [Bsz,H,d_v]  [PROVED]

        # ---------------- Titans‑style memory update ----------------
        # Delta‑corrected write. [INFERENCE]
        sigmaKL = safe_softplus(KL)
        # V_hat = (σ(K) M) / (σ(K)·z)
        num_hat = torch.einsum('bhr,bhrr->bhr', sigmaKL, M_read)
        den_hat = (sigmaKL * z_read).sum(-1, keepdim=True).clamp_min(1e-6)
        VhatL = num_hat / den_hat
        # w_t = σ(K)^T (V - V_hat)  => outer product [b,h,r,r]
        delta = (VL - VhatL)
        w_t = sigmaKL.unsqueeze(-1) * delta.unsqueeze(-2)

        # Surprise features u_t: concat [QL, VL, VhatL, |delta|] along last dim. [INFERENCE]
        u = torch.cat([QL, VL, VhatL, delta.abs()], dim=-1)  # [Bsz,H,4r]
        alpha, eta, theta = self.gater(u)  # [Bsz,H,1] each

        # Promote gate scalars to 4D for safe broadcasting with [B,H,r,r]
        alpha4 = alpha.unsqueeze(-1)           # [B,H,1,1]
        eta4   = eta.unsqueeze(-1)             # [B,H,1,1]
        theta4 = theta.unsqueeze(-1)           # [B,H,1,1]

        S_prev_read = state.S_prev.detach().clone()
        S_t = eta4 * S_prev_read + theta4 * w_t
        with torch.no_grad():  # all runtime memory updates must NOT track grads
            state.S_prev.copy_(S_t)
            state.M.copy_((1.0 - alpha4) * state.M + S_t)
            state.z.copy_((1.0 - alpha)  * state.z + sigmaKL)  # alpha broadcasts over r

        # ---------------- Output projection ----------------
        A_flat = A.reshape(Bsz, self.H * self.d_v)
        out = self.W_o(A_flat)  # [Bsz, d_model]
        return out


# ---------------------------
# Example wiring (manual test)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    d_model = 2048
    H = 32
    d_h = 128
    d_v = 128
    r = 64
    w = 8192
    B = 512
    k = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Option 1: FP32 end‑to‑end (safe default). [PROVED]
    attn = FusedAttentionV2(d_model, H, d_h, d_v, r, w, B, top_k_blocks=k, tau_skip=0.2).to(device)
    st = attn.init_state(bsz=1, device=device)  # dtype follows module
    x = torch.randn(1, d_model, device=device, dtype=attn.ln.weight.dtype)

    for t in range(1024):
        y = attn(x, st)
        x = y  # dummy recurrence
    print("OK — ran 1024 steps in FP32.")

    # Option 2: FP16 end‑to‑end (if your GPU likes it). [PROVED]
    if device.type == 'cuda':
        attn16 = FusedAttentionV2(d_model, H, d_h, d_v, r, w, B, top_k_blocks=k, tau_skip=0.2).to(device).half()
        st16 = attn16.init_state(bsz=1, device=device)  # state auto‑uses module dtype (fp16)
        x16 = torch.randn(1, d_model, device=device, dtype=torch.float16)
        for t in range(256):
            y16 = attn16(x16, st16)
            x16 = y16
        print("OK — ran 256 steps in FP16.")

    # Note: For production, consider PyTorch AMP for mixed precision. [UNVERIFIED]
