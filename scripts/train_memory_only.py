# ==============================================================
# Windows-friendly, Triton-free training script
# - No Triton kernels
# - No external fused_attn_v2 import (re-implemented below)
# - torch.compile disabled by default for maximum compatibility
# - Includes BOTH: (1) Vanilla MHA baseline, (2) Fused-like latent attention
# ==============================================================

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

# ----------------------- Setup & device -----------------------
print("Starting training script...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA required."
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.manual_seed(0)

dev = torch.device("cuda:0")
use_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
dty = torch.bfloat16 if use_bf16 else torch.float16
amp_dtype = dty
USE_COMPILE = False  # keep False for Windows stability

# ----------------------- Task & curriculum --------------------
V, ASK, FILL, GAP = 256, 0, 1, 128
V_TARGET, GAP_TARGET = 256, 128  # Keep targets same as initial values to prevent curriculum
BUMP_STEPS = []  # No curriculum steps needed
TOTAL_STEPS_FUSED = 500  # Total epochs to train

# Distractor configuration / ramps
DISTRACTOR_P_BASE = 0.55
DISTRACTOR_P_CUR = DISTRACTOR_P_BASE
RAMP_ACTIVE = False
RAMP_LEN = 200
_ramp_t = 0
_ramp_start = DISTRACTOR_P_BASE
_ramp_target = DISTRACTOR_P_BASE

# ----------------------- Global train knobs -------------------
BASE_LR = 0.00153
LR_WARMUP_STEPS_PER_PHASE = 100
WARMUP_START_FACTOR = 0.50
SMOOTH = 0.02  # CE smoothing
GRAD_CLIP = 0.5

# ==============================================================
# Fused-like attention module (pure PyTorch re-implementation)
# ==============================================================

class FusedV2Config:
    def __init__(
        self,
        d_model=32, n_heads=4, head_dim=8, d_value=8,
        r_latent=16, window=32, block_size=8,
        top_k_blocks=2, tau_skip=-1.0, bsz=1,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_value = d_value
        self.r_latent = r_latent
        self.window = window
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.tau_skip = tau_skip
        self.bsz = bsz

class TinyRouter(nn.Module):
    def __init__(self, in_dim, hidden, device, dtype):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden, 1, device=device, dtype=dtype)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TitansGater(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, device, dtype):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden, out_dim, device=device, dtype=dtype)
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))

class FusedAttentionV2(nn.Module):
    """
    Parameter layout mirrors the original fused module so the training loop
    (manual Q/K/V projections into latent space + online stats) works unchanged.
    """
    def __init__(self, cfg: FusedV2Config, device, dtype):
        super().__init__()
        H, d_h, d_v, r = cfg.n_heads, cfg.head_dim, cfg.d_value, cfg.r_latent
        d_model = cfg.d_model

        # Keep LN in model's dtype; we cast inputs in the trainer if needed
        self.ln = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True, device=device, dtype=dtype)
        self.W_qkv = nn.Linear(d_model, H * (2 * d_h + d_v), bias=False, device=device, dtype=dtype)
        self.W_o   = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        # Latent projections
        self.E_q = nn.Parameter(torch.empty(H, d_h, r, device=device, dtype=dtype))
        self.E_k = nn.Parameter(torch.empty(H, d_h, r, device=device, dtype=dtype))
        self.E_v = nn.Parameter(torch.empty(H, d_v, r, device=device, dtype=dtype))
        self.D_v = nn.Parameter(torch.empty(H, r, d_v, device=device, dtype=dtype))

        # Mixture / gating knobs
        self.beta = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.router = TinyRouter(in_dim=r, hidden=2*r, device=device, dtype=dtype)
        self.gater  = TitansGater(in_dim=2*d_h, hidden=2*d_h, out_dim=(2*d_h + d_v), device=device, dtype=dtype)

        # Bookkeeping
        self.H = H
        self.d_h = d_h
        self.d_v = d_v
        self.d_model = d_model
        self.r = r
        self.k_top = 4
        self.tau_skip = cfg.tau_skip

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        for p in [self.E_q, self.E_k, self.E_v, self.D_v]:
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x):
        # Unused in training path (trainer does explicit latent operations).
        # Define a simple pass to make printing / optional usage safe.
        x = self.ln(x)
        return self.W_o(x)

def build_fused_v2(cfg: FusedV2Config):
    device = dev
    dtype = dty
    attn = FusedAttentionV2(cfg, device=device, dtype=dtype)
    with torch.no_grad():
        assert cfg.n_heads * cfg.d_value == cfg.d_model
        attn.W_o.weight.copy_(torch.eye(cfg.d_model, device=device, dtype=dtype))
    st = attn.state_dict()
    return attn, st, device, dtype

# ==============================================================
# Vanilla MHA baseline (pure PyTorch, causal, no Triton)
# ==============================================================

class VanillaMHA(nn.Module):
    """
    Standard scaled dot-product self-attention w/ pre-LN and output proj.
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int, device, dtype):
        super().__init__()
        assert d_model == n_heads * head_dim, "d_model must equal n_heads * head_dim"
        self.d_model  = d_model
        self.H        = n_heads
        self.d_h      = head_dim
        self.scale    = 1.0 / math.sqrt(head_dim)

        # LN in fp32 for numerical stability on Windows
        self.ln = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True)
        self.ln.to(device=device, dtype=torch.float32)

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)
        self.W_o   = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x):  # x: (B,T,D)
        B, T, D = x.shape
        x_fp32 = x.to(torch.float32)
        x_fp32 = self.ln(x_fp32)
        x = x_fp32.to(x.dtype)

        qkv = self.W_qkv(x).view(B, T, self.H, 3 * self.d_h)
        Q, K, V = torch.split(qkv, [self.d_h, self.d_h, self.d_h], dim=-1)  # (B,T,H,d_h)
        Q = Q.permute(0, 2, 1, 3)  # (B,H,T,d_h)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
        attn_scores.masked_fill_(causal_mask, float('-inf'))
        A = attn_scores.softmax(dim=-1)  # (B,H,T,T)

        ctx = torch.matmul(A, V)  # (B,H,T,d_h)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # (B,T,D)
        return self.W_o(ctx)

# ==============================================================
# Shared utils / data generator
# ==============================================================

def gpu_mem(msg=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserv = torch.cuda.memory_reserved() / (1024**2)
    print(f"[mem] {msg} allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

def ce_smooth(logits, target, smoothing=0.0):
    logits_f32 = logits.float()
    if smoothing <= 0.0:
        return F.cross_entropy(logits_f32, target)
    n = logits_f32.size(-1)
    logp = F.log_softmax(logits_f32, dim=-1)
    with torch.no_grad():
        dist = torch.full_like(logp, smoothing / (n - 1))
        dist.scatter_(1, target.view(-1, 1), 1.0 - smoothing)
    return -(dist * logp).sum(dim=-1).mean()

def start_distractor_ramp(p_start=0.45, p_target=DISTRACTOR_P_BASE, length=300):
    global RAMP_ACTIVE, _ramp_t, _ramp_start, _ramp_target, RAMP_LEN, DISTRACTOR_P_CUR
    _ramp_start = float(p_start)
    _ramp_target = float(p_target)
    RAMP_LEN = int(max(1, length))
    _ramp_t = 0
    RAMP_ACTIVE = True
    DISTRACTOR_P_CUR = _ramp_start
    print(f"[ramp] distractor_p from {p_start:.2f} -> {p_target:.2f} over {RAMP_LEN} steps")

def tick_distractor_ramp():
    global RAMP_ACTIVE, _ramp_t, DISTRACTOR_P_CUR
    if not RAMP_ACTIVE:
        return
    _ramp_t += 1
    if _ramp_t >= RAMP_LEN:
        DISTRACTOR_P_CUR = _ramp_target
        RAMP_ACTIVE = False
    else:
        alpha = _ramp_t / RAMP_LEN
        DISTRACTOR_P_CUR = _ramp_start + alpha * (_ramp_target - _ramp_start)

def build_sequences(B):
    low = max(2, GAP // 2)
    high = GAP + 1
    Gs = torch.randint(low=low, high=high, size=(B,), device=dev)
    T_max = int((Gs + 2).max().item())

    keys = torch.randint(2, V, (B,), device=dev)
    seq  = torch.full((B, T_max), ASK, device=dev, dtype=torch.long)
    seq[:, 0] = keys

    if T_max > 2:
        distr = torch.randint(2, V, (B, T_max - 2), device=dev)
        key_cols = keys.unsqueeze(1).expand_as(distr)
        mask_eq = (distr == key_cols)
        if mask_eq.any():
            distr_alt = 2 + ((distr - 2 + 1) % (V - 2))
            distr = torch.where(mask_eq, distr_alt, distr)

        chooser = (torch.rand(B, T_max - 2, device=dev) < DISTRACTOR_P_CUR)
        fillers = torch.where(chooser, distr, torch.full_like(distr, FILL))

        for i in range(B):
            g = int(Gs[i].item())
            if g > 0:
                seq[i, 1:1+g] = fillers[i, :g]

    return seq, keys, Gs, T_max

# ==============================================================
# Configs
# ==============================================================

cfg = FusedV2Config(
    d_model=32, n_heads=4, head_dim=8, d_value=8,
    r_latent=16, window=32, block_size=8,
    top_k_blocks=2, tau_skip=-1.0, bsz=1,
)

# ==============================================================
# ----------- Baseline MHA: training & eval (Windows OK) -------
# ==============================================================

RUN_MHA = True
BATCH_MHA = 4096
TOTAL_STEPS_MHA = 2600
EVAL_EVERY_MHA = 500
EVAL_EPISODES_MHA = 5000
LR_MHA = 6e-4
WD_MHA = 0.10
SMOOTH_MHA = 0.0

attn_mha = VanillaMHA(cfg.d_model, cfg.n_heads, cfg.head_dim, device=dev, dtype=dty)
emb_mha  = nn.Embedding(V, cfg.d_model, device=dev, dtype=dty)
head_mha = nn.Linear(cfg.d_model, V, bias=True, device=dev, dtype=dty)

if USE_COMPILE:
    try:
        attn_mha = torch.compile(attn_mha, mode="reduce-overhead", backend="inductor", fullgraph=False)
        print("Compiled VanillaMHA with torch.compile")
    except Exception as e:
        print(f"torch.compile skipped for VanillaMHA: {e}")

optimizer_mha = torch.optim.AdamW(
    list(attn_mha.parameters()) + list(emb_mha.parameters()) + list(head_mha.parameters()),
    lr=LR_MHA, betas=(0.9, 0.95), weight_decay=WD_MHA
)

def lr_lambda_mha(step):
    warm = 200
    if step < warm:
        return float(step) / float(max(1, warm))
    return 1.0

scheduler_mha = torch.optim.lr_scheduler.LambdaLR(optimizer_mha, lr_lambda_mha)
# Initialize GradScaler with enabled=False when using BFloat16 to avoid unsupported operations
scaler_mha = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

def run_batch_mha(B=4096, smoothing=SMOOTH_MHA):
    seq, keys, Gs, T = build_sequences(B)
    x = emb_mha(seq)                       # (B, T, d_model)
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        y_all = attn_mha(x)                # (B, T, d_model)
        idx = (Gs + 1).to(torch.long)
        Y = y_all[torch.arange(B, device=dev), idx]  # (B, d_model)
        logits = head_mha(Y)               # (B, V)
        loss = ce_smooth(logits, keys, smoothing=smoothing)
    return loss

@torch.no_grad()
def eval_many_mha(n_episodes=1500, B=8192):
    total = 0
    tot_ce = 0.0
    tot_acc = 0.0
    remaining = n_episodes
    attn_mha.eval()
    while remaining > 0:
        b = min(B, remaining)
        seq, keys, Gs, T = build_sequences(b)
        x = emb_mha(seq)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            y_all = attn_mha(x)
            idx = (Gs + 1).to(torch.long)
            Y = y_all[torch.arange(b, device=dev), idx]
            logits = head_mha(Y)
        ce = F.cross_entropy(logits.float(), keys).item()
        acc = (logits.argmax(-1) == keys).float().mean().item()
        tot_ce += ce * b
        tot_acc += acc * b
        total += b
        remaining -= b
    attn_mha.train()
    return tot_ce / total, tot_acc / total

def ok_to_bump_V_mha():
    ce, acc = eval_many_mha(800, B=8192)
    base = math.log(V)
    return (ce < 0.62 * base) and (acc > 0.24)

def ok_to_bump_GAP_mha():
    ce, acc = eval_many_mha(800, B=8192)
    base = math.log(V)
    return (ce < 0.72 * base) and (acc > 0.20)

def maybe_resize_vocab_mha(new_V, current_step):
    global V, emb_mha, head_mha, optimizer_mha, scheduler_mha
    if new_V <= V:
        return
    print(f"[curriculum/MHA] resizing vocab: {V} -> {new_V}")
    old_V = V
    V = new_V
    new_emb = nn.Embedding(V, cfg.d_model, device=dev, dtype=dty)
    with torch.no_grad():
        new_emb.weight[:old_V].copy_(emb_mha.weight[:old_V])
    emb_mha = new_emb
    new_head = nn.Linear(cfg.d_model, V, bias=True, device=dev, dtype=dty)
    with torch.no_grad():
        new_head.weight[:old_V].copy_(head_mha.weight[:old_V])
        new_head.bias[:old_V].copy_(head_mha.bias[:old_V])
    head_mha = new_head
    optimizer_mha = torch.optim.AdamW(
        list(attn_mha.parameters()) + list(emb_mha.parameters()) + list(head_mha.parameters()),
        lr=LR_MHA, betas=(0.9, 0.95), weight_decay=WD_MHA
    )
    scheduler_mha = torch.optim.lr_scheduler.LambdaLR(optimizer_mha, lr_lambda_mha)

def do_bump_mha(ep):
    global GAP
    scheduled = ep in BUMP_STEPS
    retry = (ep % 50 == 0) and (len(BUMP_STEPS) > 0 and ep > min(BUMP_STEPS))
    if not (scheduled or retry):
        return
    bumped = False
    if V < V_TARGET and ok_to_bump_V_mha():
        new_V = min(V * 2, V_TARGET)
        maybe_resize_vocab_mha(new_V, current_step=ep)
        bumped = True
        start_distractor_ramp(p_start=0.30, p_target=DISTRACTOR_P_BASE, length=400)
        print(f"[curriculum/MHA] V={V}, GAP={GAP} (ep={ep}, V bump)")
    if (not bumped) and (GAP < GAP_TARGET) and ok_to_bump_GAP_mha():
        GAP = min(GAP * 2, GAP_TARGET)
        bumped = True
        start_distractor_ramp(p_start=0.50, p_target=DISTRACTOR_P_BASE, length=300)
        print(f"[curriculum/MHA] V={V}, GAP={GAP} (ep={ep}, GAP bump)")

if RUN_MHA:
    print("\n=== Starting MHA training (Windows-safe, no Triton) ===")
    print(f"Device={dev}  dtype={amp_dtype}  BATCH={BATCH_MHA}  LR={LR_MHA}")
    gpu_mem("MHA: after build")

    ema_mha = None
    start_time = time.time()
    for ep in range(1, TOTAL_STEPS_MHA + 1):
        do_bump_mha(ep)
        tick_distractor_ramp()

        optimizer_mha.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss = run_batch_mha(B=BATCH_MHA, smoothing=SMOOTH_MHA)
        
        if amp_dtype == torch.float16:
            scaler_mha.scale(loss).backward()
            scaler_mha.unscale_(optimizer_mha)
            torch.nn.utils.clip_grad_norm_(
                list(attn_mha.parameters()) + list(emb_mha.parameters()) + list(head_mha.parameters()), GRAD_CLIP
            )
            scaler_mha.step(optimizer_mha)
            scaler_mha.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(attn_mha.parameters()) + list(emb_mha.parameters()) + list(head_mha.parameters()), GRAD_CLIP
            )
            optimizer_mha.step()
        
        scheduler_mha.step()

        L = loss.item()
        ema_mha = L if ema_mha is None else 0.98 * ema_mha + 0.02 * L
        if ep % 100 == 0:
            base = math.log(V)
            lr = optimizer_mha.param_groups[0]['lr']
            print(f"[MHA ep {ep:4d}] train_CE={L:.3f}  EMA={ema_mha:.3f}  CE/lnV={L/base:.3f}  ln(V)={base:.3f}  lr={lr:.2e}  V={V} GAP={GAP}")

        if ep % EVAL_EVERY_MHA == 0:
            ce, acc = eval_many_mha(EVAL_EPISODES_MHA, B=8192)
            ppl = math.exp(ce)
            n = EVAL_EPISODES_MHA
            se = (acc * (1 - acc) / n) ** 0.5
            lo = max(0.0, acc - 1.96 * se); hi = min(1.0, acc + 1.96 * se)
            print(f"[MHA]        eval_CE={ce:.3f}  acc={acc:.3f}  95%CI=[{lo:.3f},{hi:.3f}]  PPL={ppl:.1f}  baseline_PPL={V}")
            gpu_mem(f"MHA: after eval ep={ep}")
            torch.cuda.empty_cache()

    print(f"[MHA] Uniform ln(V) = {math.log(V):.3f}")
    gpu_mem("MHA: final")

# ==============================================================
# ----------- Fused-like latent attention: train & eval --------
# ==============================================================

RUN_FUSED = True

print("\nBuilding fused-like model (pure PyTorch)...")
attn, st, dev, dty = build_fused_v2(cfg)
print(f"Model built on device: {dev}")
print(attn)
attn_raw = attn

# Init & warmups
beta_start, beta_target = -2.0, -0.40
tau_start,  tau_target  =  1.0,  0.30
param_warmup_steps = 200

attn_raw.tau_skip = tau_start
with torch.no_grad():
    attn_raw.beta.data.fill_(beta_start)
    attn_raw.E_q.data.copy_(attn_raw.E_k.data)
    for h in range(attn_raw.H):
        Ev = attn_raw.E_v[h].to(torch.float32)
        attn_raw.D_v.data[h].copy_(torch.linalg.pinv(Ev).to(attn_raw.E_v.dtype))

evdv_params = [attn_raw.E_v, attn_raw.D_v]
for p in evdv_params:
    p.requires_grad_(False)

with torch.no_grad():
    assert attn_raw.H * attn_raw.d_v == attn_raw.d_model
    attn_raw.W_o.weight.copy_(torch.eye(attn_raw.d_model, device=dev, dtype=dty))
attn_raw.W_o.weight.requires_grad_(False)

# Separate embedding/head for fused path
emb  = nn.Embedding(V, cfg.d_model, device=dev, dtype=dty)
head = nn.Linear(cfg.d_model, V, bias=True, device=dev, dtype=dty)

def collect_train_params():
    params = list(emb.parameters())
    for n, p in attn_raw.named_parameters():
        if any(n.startswith(k) for k in ["W_qkv", "ln", "E_q", "E_k"]):
            params.append(p)
    params += list(head.parameters())
    return params

train_params = collect_train_params()

def _linear_warmup(opt, warm):
    try:
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_steps=warm
        )
    except TypeError:
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=warm
        )

def build_phase_scheduler(opt, current_step):
    next_bumps = [b for b in BUMP_STEPS if b > current_step]
    phase_end = min(next_bumps) if next_bumps else TOTAL_STEPS_FUSED
    phase_len = max(1, phase_end - current_step)
    warm = min(LR_WARMUP_STEPS_PER_PHASE, phase_len)
    cos_len = max(0, phase_len - warm)
    warmup = _linear_warmup(opt, warm)
    if cos_len < 200:
        hold = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=max(1, cos_len))
        return torch.optim.lr_scheduler.SequentialLR(opt, [warmup, hold], milestones=[warm])
    else:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cos_len, eta_min=2e-4)
        return torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warm])

def build_optimizer_and_scheduler(current_step):
    global train_params
    train_params = collect_train_params()
    opt = torch.optim.AdamW(train_params, lr=BASE_LR, weight_decay=0.01)
    opt.add_param_group({'params': evdv_params, 'lr': BASE_LR * 0.33, 'weight_decay': 0.01})
    sched = build_phase_scheduler(opt, current_step)
    return opt, sched

opt, sched = build_optimizer_and_scheduler(current_step=1)
# Initialize GradScaler with enabled=False when using BFloat16 to avoid unsupported operations
scaler_fused = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
softplus = F.softplus

if USE_COMPILE:
    try:
        attn = torch.compile(attn, mode="reduce-overhead", backend="inductor", fullgraph=False)
        print("Compiled fused module with torch.compile")
        attn_raw = attn
    except Exception as e:
        print(f"torch.compile skipped: {e}")

# Projections
def batched_proj_qkv(x):  # x: (B,T,d_model)
    x   = attn_raw.ln(x)
    qkv = attn_raw.W_qkv(x).view(x.size(0), x.size(1), attn_raw.H, 2*attn_raw.d_h + attn_raw.d_v)
    Q, K, Vv = torch.split(qkv, [attn_raw.d_h, attn_raw.d_h, attn_raw.d_v], dim=-1)
    QL = torch.einsum('bthd,hdr->bthr', Q,  attn_raw.E_q)
    KL = torch.einsum('bthd,hdr->bthr', K,  attn_raw.E_k)
    VL = torch.einsum('bthd,hdr->bthr', Vv, attn_raw.E_v)
    return QL, KL, VL

# Vectorized train/eval
def run_batch_fused(B=4096, smoothing=SMOOTH):
    seq, keys, Gs, T = build_sequences(B)
    x   = emb(seq)
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        QL, KL, VL = batched_proj_qkv(x)
        sK = softplus(KL)

        M = torch.zeros(B, attn_raw.H, attn_raw.r, attn_raw.r, device=dev, dtype=dty)
        z = torch.full((B, attn_raw.H, attn_raw.r), 1e-3, device=dev, dtype=dty)

        for t in range(T - 1):
            active = (t < (Gs + 1)).to(dtype=dty).view(B, 1, 1)
            sKt = sK[:, t] * active
            VLt = VL[:, t] * active
            num_hat = (sKt.unsqueeze(-2) @ M).squeeze(-2)
            den_hat = (sKt * z).sum(-1, keepdim=True).clamp_min(1e-6)
            VhatL   = num_hat / den_hat
            w = sKt.unsqueeze(-1) * (VLt - VhatL).unsqueeze(-2)
            M = M + w
            z = z + sKt

        idx = (Gs + 1).to(torch.long)
        sQ  = softplus(QL[torch.arange(B, device=dev), idx])
        num = (sQ.unsqueeze(-2) @ M).squeeze(-2)
        den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
        YL  = num / den
        Y   = torch.einsum('bhr,hrd->bhd', YL, attn_raw.D_v)

        logits = head(Y.reshape(B, -1))
        loss = ce_smooth(logits, keys, smoothing=smoothing)
    return loss

@torch.no_grad()
def eval_many_fused(n_episodes=1500, B=8192):
    total = 0
    tot_ce = 0.0
    tot_acc = 0.0
    remaining = n_episodes
    while remaining > 0:
        b = min(B, remaining)
        seq, keys, Gs, T = build_sequences(b)
        x   = emb(seq)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            QL, KL, VL = batched_proj_qkv(x)
            sK = softplus(KL)
            M = torch.zeros(b, attn_raw.H, attn_raw.r, attn_raw.r, device=dev, dtype=dty)
            z = torch.full((b, attn_raw.H, attn_raw.r), 1e-3, device=dev, dtype=dty)
            for t in range(T - 1):
                active = (t < (Gs + 1)).to(dtype=dty).view(b, 1, 1)
                sKt = sK[:, t] * active
                VLt = VL[:, t] * active
                num_hat = (sKt.unsqueeze(-2) @ M).squeeze(-2)
                den_hat = (sKt * z).sum(-1, keepdim=True).clamp_min(1e-6)
                VhatL   = num_hat / den_hat
                w = sKt.unsqueeze(-1) * (VLt - VhatL).unsqueeze(-2)
                M = M + w
                z = z + sKt
            idx = (Gs + 1).to(torch.long)
            sQ  = softplus(QL[torch.arange(b, device=dev), idx])
            num = (sQ.unsqueeze(-2) @ M).squeeze(-2)
            den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
            YL  = num / den
            Y   = torch.einsum('bhr,hrd->bhd', YL, attn_raw.D_v)
            logits = head(Y.reshape(b, -1))
        ce = F.cross_entropy(logits.float(), keys).item()
        acc = (logits.argmax(-1) == keys).float().mean().item()
        tot_ce += ce * b
        tot_acc += acc * b
        total += b
        remaining -= b
    return tot_ce / total, tot_acc / total

EXTRA_BUMP_CHECK_EVERY = 50
ktop_override_steps = 0  # temporarily force higher k_top after V bump

def ok_to_bump_V_fused():
    ce, acc = eval_many_fused(800, B=8192)
    base = math.log(V)
    return (ce < 0.62 * base) and (acc > 0.24)

def ok_to_bump_GAP_fused():
    ce, acc = eval_many_fused(800, B=8192)
    base = math.log(V)
    return (ce < 0.72 * base) and (acc > 0.20)

def maybe_resize_vocab_fused(new_V, current_step):
    global V, emb, head, opt, sched, train_params
    if new_V <= V:
        return
    print(f"[curriculum] resizing vocab: {V} -> {new_V}")
    old_V = V
    V = new_V

    new_emb = nn.Embedding(V, cfg.d_model, device=dev, dtype=dty)
    with torch.no_grad():
        new_emb.weight[:old_V].copy_(emb.weight[:old_V])
    emb = new_emb

    new_head = nn.Linear(cfg.d_model, V, bias=True, device=dev, dtype=dty)
    with torch.no_grad():
        new_head.weight[:old_V].copy_(head.weight[:old_V])
        new_head.bias[:old_V].copy_(head.bias[:old_V])
    head = new_head

    opt, sched = build_optimizer_and_scheduler(current_step=current_step)
    return opt, sched

def do_bump_fused(ep):
    global GAP, opt, sched, ktop_override_steps
    scheduled = ep in BUMP_STEPS
    retry     = (ep % EXTRA_BUMP_CHECK_EVERY == 0) and (ep > min(BUMP_STEPS))
    if not (scheduled or retry):
        return
    bumped = False
    if V < V_TARGET and ok_to_bump_V_fused():
        new_V = min(V * 2, V_TARGET)
        opt, sched = maybe_resize_vocab_fused(new_V, current_step=ep) or (opt, sched)
        bumped = True
        start_distractor_ramp(p_start=0.30, p_target=DISTRACTOR_P_BASE, length=400)
        ktop_override_steps = max(ktop_override_steps, 300)  # force k_top=5 for ~300 steps
        print(f"[curriculum] V={V}, GAP={GAP} (ep={ep}, V bump)")
    if (not bumped) and (GAP < GAP_TARGET) and ok_to_bump_GAP_fused():
        GAP = min(GAP * 2, GAP_TARGET)
        bumped = True
        start_distractor_ramp(p_start=0.50, p_target=DISTRACTOR_P_BASE, length=300)
        print(f"[curriculum] V={V}, GAP={GAP} (ep={ep}, GAP bump)")

if RUN_FUSED:
    ema = None
    evdv_unfrozen = False
    BATCH = 4096
    EVAL_BATCH = 8192
    EVAL_EVERY = 500
    EVAL_EPISODES = 5000

    gpu_mem("fused: after build")

    for ep in range(1, TOTAL_STEPS_FUSED + 1):
        # Curriculum
        do_bump_fused(ep)

        # Unfreeze E_v/D_v after warmup
        if not evdv_unfrozen and ep > param_warmup_steps:
            for p in evdv_params:
                p.requires_grad_(True)
            evdv_unfrozen = True

        # Beta & tau schedules
        if ep <= param_warmup_steps:
            frac = 0.0
        else:
            frac = min(1.0, (ep - param_warmup_steps) / 2000)
        cosf = 0.5 * (1 - math.cos(math.pi * frac))
        with torch.no_grad():
            attn_raw.beta.data.fill_(beta_start + cosf * (beta_target - beta_start))
            attn_raw.tau_skip = tau_start + cosf * (tau_target - tau_start)
            if ep > param_warmup_steps:
                attn_raw.k_top = 5 if GAP >= 64 else 4
            else:
                if ktop_override_steps > 0:
                    attn_raw.k_top = 5
                else:
                    attn_raw.k_top = 4

        if ktop_override_steps > 0:
            ktop_override_steps -= 1

        tick_distractor_ramp()

        # Train step
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss = run_batch_fused(BATCH, smoothing=SMOOTH)
        
        if amp_dtype == torch.float16:
            scaler_fused.scale(loss).backward()
            scaler_fused.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(collect_train_params(), GRAD_CLIP)
            scaler_fused.step(opt)
            scaler_fused.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(collect_train_params(), GRAD_CLIP)
            opt.step()
            
        sched.step()

        # Logs
        L = loss.item()
        ema = L if ema is None else 0.98 * ema + 0.02 * L
        if ep % 100 == 0:
            mix = torch.sigmoid(attn_raw.beta).mean().item()
            base = math.log(V)
            lr = opt.param_groups[0]['lr']
            print(f"[ep {ep:4d}] train_CE={L:.3f}  EMA={ema:.3f}  CE/lnV={L/base:.3f}  ln(V)={base:.3f}  mix={mix:.3f}  lr={lr:.2e}  V={V} GAP={GAP}")

        if ep % EVAL_EVERY == 0:
            ce, acc = eval_many_fused(EVAL_EPISODES, B=EVAL_BATCH)
            ppl = math.exp(ce)
            n = EVAL_EPISODES
            se = (acc * (1 - acc) / n) ** 0.5
            lo = max(0.0, acc - 1.96 * se); hi = min(1.0, acc + 1.96 * se)
            print(f"           eval_CE={ce:.3f}  acc={acc:.3f}  95%CI=[{lo:.3f},{hi:.3f}]  PPL={ppl:.1f}  baseline_PPL={V}")
            gpu_mem(f"fused: after eval ep={ep}")
            torch.cuda.empty_cache()

    print(f"Uniform ln(V) = {math.log(V):.3f}")
    gpu_mem("fused: final")
