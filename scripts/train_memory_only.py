# train_memory_only_grad_linear_min.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from fused_attn_v2 import FusedV2Config, build_fused_v2

assert torch.cuda.is_available(), "CUDA required."

# Start with easier task (V=128, GAP=32), scale up later
V, ASK, FILL, GAP = 128, 0, 1, 32
V_TARGET, GAP_TARGET = 512, 128  # Target difficulty
cfg = FusedV2Config(d_model=512, n_heads=8, head_dim=64, d_value=64,
                    r_latent=32, window=64, block_size=8, top_k_blocks=2,
                    tau_skip=-1.0, bsz=1)
attn, st, dev, dty = build_fused_v2(cfg)
torch.manual_seed(0)

# Warm-up hyper-params
warmup_steps = 2000   # purely local path (extended for stability)
beta_start = -3.0     # σ≈0.05 (~95% local)
beta_target = -0.85   # σ≈0.3 mid-mix

tau_start = 1.0
tau_target = 0.3

attn.tau_skip = tau_start
with torch.no_grad():
    attn.beta.data.fill_(beta_start)  # strongly local mix
    # Stable latent init: E_q matches E_k initially
    attn.E_q.data.copy_(attn.E_k.data)
    # Initialize D_v as pseudo-inverse of E_v
    for h in range(attn.H):
        Ev = attn.E_v[h].to(torch.float32)
        attn.D_v.data[h].copy_(torch.linalg.pinv(Ev).to(attn.E_v.dtype))
# Freeze E_v & D_v during warm-up; unfreeze later for fine-tuning
evdv_params = [attn.E_v, attn.D_v]
for p in evdv_params:
    p.requires_grad_(False)

# Identity W_o (since H*d_v == d_model == 512)
with torch.no_grad():
    assert attn.H * attn.d_v == attn.d_model
    attn.W_o.weight.copy_(torch.eye(attn.d_model, device=dev, dtype=dty))
attn.W_o.weight.requires_grad_(False)

emb  = nn.Embedding(V, cfg.d_model, device=dev, dtype=dty)
head = nn.Linear(cfg.d_model, V, bias=True, device=dev, dtype=dty)

# Train emb + W_qkv + ln + head
train_params = list(emb.parameters())
for n,p in attn.named_parameters():
    if any(n.startswith(k) for k in ["W_qkv", "ln", "E_q", "E_k"]):
        train_params.append(p)
train_params += list(head.parameters())

opt = torch.optim.AdamW(train_params, lr=3e-4, weight_decay=0.01)
opt.add_param_group({'params': evdv_params, 'lr': 1e-4, 'weight_decay': 0.01})
train_params += evdv_params
# eta_min should be lower than base lr
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=3e-5)
softplus = F.softplus

def ce_smooth(logits, target, smoothing=0.0):
    if smoothing <= 0.0:
        return F.cross_entropy(logits, target)
    n = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        dist = torch.full_like(logp, smoothing / (n - 1))
        dist.scatter_(1, target.view(-1, 1), 1.0 - smoothing)
    return -(dist * logp).sum(dim=-1).mean()

def proj_qkv(x):
    x = attn.ln(x)
    qkv = attn.W_qkv(x).view(1, attn.H, 2*attn.d_h + attn.d_v)
    Q, K, Vv = torch.split(qkv, [attn.d_h, attn.d_h, attn.d_v], dim=-1)
    QL = torch.einsum('bhd,hdr->bhr', Q, attn.E_q)
    KL = torch.einsum('bhd,hdr->bhr', K, attn.E_k)
    VL = torch.einsum('bhd,hdr->bhr', Vv, attn.E_v)
    return QL, KL, VL

@torch.no_grad()
def eval_many(n_episodes=200):
    H, r = attn.H, attn.r
    tot_ce, acc = 0.0, 0
    eps = 1e-3
    for _ in range(n_episodes):
        M = torch.zeros(1, H, r, r, device=dev, dtype=dty); z = torch.full((1, H, r), eps, device=dev, dtype=dty)
        key = torch.randint(2, V, (1,), device=dev).item()
        seq = [key] + [FILL]*GAP + [ASK]
        for tok in seq[:-1]:
            QL, KL, VL = proj_qkv(emb(torch.tensor([tok], device=dev)))
            sK = softplus(KL)
            # Delta-corrected Titans memory update
            num_hat = torch.einsum('bhr,bhrr->bhr', sK, M)
            den_hat = (sK * z).sum(-1, keepdim=True).clamp_min(1e-6)
            VhatL = num_hat / den_hat
            w = sK.unsqueeze(-1) * (VL - VhatL).unsqueeze(-2)
            M.add_(w)
            z.add_(sK)
        QLq, _, _ = proj_qkv(emb(torch.tensor([ASK], device=dev)))
        sQ = softplus(QLq)
        num = torch.einsum('bhr,bhrr->bhr', sQ, M)
        den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
        YL  = num / den
        Y   = torch.einsum('bhr,hrd->bhd', YL, attn.D_v)    # decode
        logits = head(Y.reshape(1, -1))                     # W_o is identity
        tgt = torch.tensor([key], device=dev)
        tot_ce += F.cross_entropy(logits, tgt).item()
        acc += int(logits.argmax(-1).item() == key)
    return tot_ce/n_episodes, acc/n_episodes

def run_episode():
    H, r = attn.H, attn.r
    eps = 1e-3  # Match eval stability
    M = torch.zeros(1, H, r, r, device=dev, dtype=dty)
    z = torch.full((1, H, r), eps, device=dev, dtype=dty)  # Prefill z
    key = torch.randint(2, V, (1,), device=dev).item()
    seq = [key] + [FILL]*GAP + [ASK]
    # Process sequence
    for tok in seq[:-1]:
        QL, KL, VL = proj_qkv(emb(torch.tensor([tok], device=dev)))
        sK = softplus(KL)
        # Delta-corrected update (stable with z prefilled)
        num_hat = torch.einsum('bhr,bhrr->bhr', sK, M)
        den_hat = (sK * z).sum(-1, keepdim=True).clamp_min(1e-6)
        VhatL = num_hat / den_hat
        w = sK.unsqueeze(-1) * (VL - VhatL).unsqueeze(-2)
        M = M + w
        z = z + sK
    # Query
    QLq, _, _ = proj_qkv(emb(torch.tensor([ASK], device=dev)))
    sQ = softplus(QLq)
    num = torch.einsum('bhr,bhrr->bhr', sQ, M)
    den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
    YL = num / den
    Y = torch.einsum('bhr,hrd->bhd', YL, attn.D_v)
    logits = head(Y.reshape(1, -1))
    tgt = torch.tensor([key], device=dev)
    return ce_smooth(logits, tgt, smoothing=0.1)

def run_batch(batch_size=64):
    """Run batch_size episodes and return mean loss."""
    losses = []
    for _ in range(batch_size):
        losses.append(run_episode())
    return sum(losses) / len(losses)

ema = None
evdv_unfrozen = False
batch_size = 64  # Mini-batch episodes per step

for ep in range(1, 10001):  # More steps for curriculum
    # ---- Curriculum: scale up difficulty if ready ----
    if ep % 500 == 0 and V < V_TARGET:
        ce, _ = eval_many(100)
        if ce < math.log(V) - 1.0:  # If doing well, make harder
            V = min(V * 2, V_TARGET)
            GAP = min(GAP * 2, GAP_TARGET)
            print(f"[curriculum] V={V}, GAP={GAP}")
    
    # ---- Parameter schedules ----
    if not evdv_unfrozen and ep > warmup_steps:
        for p in evdv_params:
            p.requires_grad_(True)
        evdv_unfrozen = True
    
    # Beta and tau_skip schedules
    if ep <= warmup_steps:
        frac = 0.0
    else:
        frac = min(1.0, (ep - warmup_steps) / 2000)  # 2000 steps to reach target
    
    cosf = 0.5 * (1 - math.cos(math.pi * frac))  # cosine easing
    
    with torch.no_grad():
        # Beta: start very local, relax over time
        attn.beta.data.fill_(beta_start + cosf * (beta_target - beta_start))
        # Tau_skip: start high (skip local), decrease (allow local)
        attn.tau_skip = tau_start + cosf * (tau_target - tau_start)
        # k_top: start with 1 block (local), increase to cfg.top_k_blocks
        attn.k_top = 1 if ep <= warmup_steps else cfg.top_k_blocks
    
    # ---- Training step with mini-batch episodes ----
    loss = run_batch(batch_size)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(train_params, 0.5)
    opt.step()
    sched.step()
    
    # Track loss EMA
    L = loss.item()
    ema = L if ema is None else 0.98 * ema + 0.02 * L
    
    # Logging
    if ep % 100 == 0:
        ce, acc = eval_many(200)  # Full eval
        mix = torch.sigmoid(attn.beta).mean().item()
        print(f"[ep {ep:4d}] CE={L:.3f} EMA={ema:.3f}  eval_CE={ce:.3f} acc={acc:.3f} mix={mix:.3f} V={V} GAP={GAP}")

print(f"Uniform ln(V) = {math.log(V):.3f}")
