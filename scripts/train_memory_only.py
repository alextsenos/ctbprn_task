# train_memory_only_grad_linear_min.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from fused_attn_v2 import FusedV2Config, build_fused_v2

assert torch.cuda.is_available(), "CUDA required."

V, ASK, FILL, GAP = 512, 0, 1, 128
cfg = FusedV2Config(d_model=512, n_heads=8, head_dim=64, d_value=64,
                    r_latent=32, window=64, block_size=8, top_k_blocks=2,
                    tau_skip=-1.0, bsz=1)
attn, st, dev, dty = build_fused_v2(cfg)
torch.manual_seed(0)

# Warm-up hyper-params
warmup_steps = 2000  # purely local path
beta_start = -4.0    # σ≈0.018 (~98% local)
beta_target = -0.85  # σ≈0.3 mid-mix

tau_start = 1.0
tau_target = 0.3

attn.tau_skip = tau_start
with torch.no_grad():
    attn.beta.data.fill_(beta_start)  # strongly local mix
    attn.E_q.data.copy_(attn.E_k.data)     # stable latent
    for h in range(attn.H):
        Ev = attn.E_v[h].to(torch.float32)
        attn.D_v.data[h].copy_(torch.linalg.pinv(Ev).to(attn.E_v.dtype))
# Keep E_v & D_v frozen; allow E_q/E_k to adapt latent similarity
for p in [attn.E_v, attn.D_v]:
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
    M = torch.zeros(1, H, r, r, device=dev, dtype=dty); z = torch.zeros(1, H, r, device=dev, dtype=dty)
    key = torch.randint(2, V, (1,), device=dev).item()
    seq = [key] + [FILL]*GAP + [ASK]
    for tok in seq[:-1]:
        QL, KL, VL = proj_qkv(emb(torch.tensor([tok], device=dev)))
        sK = softplus(KL)
        num_hat = torch.einsum('bhr,bhrr->bhr', sK, M)
        den_hat = (sK * z).sum(-1, keepdim=True).clamp_min(1e-6)
        VhatL = num_hat / den_hat
        w = sK.unsqueeze(-1) * (VL - VhatL).unsqueeze(-2)
        M = M + w
        z = z + sK
    QLq, _, _ = proj_qkv(emb(torch.tensor([ASK], device=dev)))
    sQ = softplus(QLq)
    num = torch.einsum('bhr,bhrr->bhr', sQ, M)
    den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
    YL  = num / den
    Y   = torch.einsum('bhr,hrd->bhd', YL, attn.D_v)
    logits = head(Y.reshape(1, -1))
    tgt = torch.tensor([key], device=dev)
    return ce_smooth(logits, tgt, smoothing=0.1)

ema = None
for ep in range(1, 2001):
    # Schedule beta and tau_skip
    if ep <= warmup_steps:
        frac = 0.0
    else:
        frac = (ep - warmup_steps) / max(1, 2000 - warmup_steps)
    # cosine easing
    cosf = 0.5 * (1 - math.cos(math.pi * frac))
    with torch.no_grad():
        attn.beta.data.fill_(beta_start + cosf * (beta_target - beta_start))
        attn.tau_skip = tau_start + cosf * (tau_target - tau_start)
        # k_top schedule: start strict local neighborhood (1) then increase to cfg value
        attn.k_top = 1 if ep <= warmup_steps else cfg.top_k_blocks
    loss = run_episode()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(train_params, 0.5)
    opt.step(); sched.step()
    L = loss.item(); ema = L if ema is None else 0.98*ema + 0.02*L
    if ep % 100 == 0:
        ce, acc = eval_many(200)
        mix = torch.sigmoid(attn.beta).mean().item()
        print(f"[ep {ep:4d}] CE={L:.3f} EMA={ema:.3f}  eval_CE={ce:.3f} acc={acc:.3f} mix={mix:.3f}")

print(f"Uniform ln(V) = {math.log(V):.3f}")
