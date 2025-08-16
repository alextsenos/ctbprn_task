import math, torch, torch.nn as nn, torch.nn.functional as F
from fused_attn_v2 import FusedV2Config, build_fused_v2

assert torch.cuda.is_available(), "CUDA required."

# --- config: tiny vocab + local-only path
V = 256  # tiny vocab so ln(V) ~ 5.545
cfg = FusedV2Config(
    d_model=128, n_heads=4, head_dim=32, d_value=32,
    r_latent=16, window=64, block_size=8, top_k_blocks=2,
    tau_skip=1.0,   # always use local path
    bsz=1,
)
attn, st, dev, dty = build_fused_v2(cfg)

# Heavily favor local path in the mixer
with torch.no_grad():
    attn.beta.data.fill_(-4.0)  # σ(-4) ≈ 0.018 → ~98% local

# tied embedding/output head
emb = nn.Embedding(V, cfg.d_model, device=dev, dtype=dty)
head = nn.Linear(cfg.d_model, V, bias=False, device=dev, dtype=dty)
head.weight = emb.weight  # weight tying

opt = torch.optim.AdamW([*emb.parameters(), *attn.parameters()], lr=3e-4)
torch.manual_seed(0)

def next_token(i):
    return (i + 1) % V

steps = 1200
# Cosine LR decay after linear warm-up
base_lr = 3e-4
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=base_lr * 0.1)
warmup_steps = 100  # first 100 steps linear warm-up
ema = None
for t in range(steps):
    tok = torch.tensor([t % V], device=dev)
    tgt = torch.tensor([next_token(t % V)], device=dev)
    y = attn(emb(tok), st)
    logits = head(y)
    loss = F.cross_entropy(logits, tgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    # LR schedule
    if t < warmup_steps:
        lr_now = base_lr * (t + 1) / warmup_steps
        for g in opt.param_groups:
            g['lr'] = lr_now
    else:
        sched.step()

    L = loss.item()
    ema = L if ema is None else 0.98 * ema + 0.02 * L
    if (t + 1) % 100 == 0:
        print(f"[{t + 1:4d}] CE={L:.3f}  EMA={ema:.3f}")

def eval_cycle_loss(attn, st, emb, head, V=256):
    # freeze state for eval
    st_snapshot = attn.init_state(bsz=1, device=emb.weight.device, dtype=emb.weight.dtype)
    with torch.no_grad():
        total = 0.0
        for j in range(V):
            tok = torch.tensor([j], device=emb.weight.device)
            tgt = torch.tensor([(j+1)%V], device=emb.weight.device)
            y = attn(emb(tok), st_snapshot)
            total += F.cross_entropy(head(y), tgt).item()
    return total / V

cycle_ce = eval_cycle_loss(attn, st, emb, head)
print("cycle CE:", cycle_ce)

# Router/mixer sanity probe
with torch.no_grad():
    x_probe = torch.randn(1, attn.d_model, device=dev, dtype=dty)
    q_probe = attn.W_qkv(attn.ln(x_probe)).view(1, attn.H, -1)[..., :attn.d_h]
    ql_probe = torch.einsum('bhd,hdr->bhr', q_probe, attn.E_q)
    p_skip = attn.router(ql_probe)
    local_ratio = (p_skip <= attn.tau_skip).float().mean().item()
    mix_sigmoid = torch.sigmoid(attn.beta).mean().item()
    print(f"router local ratio: {local_ratio:.3f}")
    print(f"mix σ(β): {mix_sigmoid:.3f}")
