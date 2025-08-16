# scripts/tune_optuna_memory_only.py
import math, argparse, optuna, torch, torch.nn as nn, torch.nn.functional as F
from optuna.pruners import SuccessiveHalvingPruner
from fused_attn_v2 import FusedV2Config, build_fused_v2

# ---- tiny helper loss (with optional label smoothing)
def ce_smooth(logits, target, smoothing=0.0):
    if smoothing <= 0.0:
        return F.cross_entropy(logits, target)
    n = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        dist = torch.full_like(logp, smoothing / (n - 1))
        dist.scatter_(1, target.view(-1, 1), 1.0 - smoothing)
    return -(dist * logp).sum(dim=-1).mean()

def build_vocab_head(d_model, V, device, dtype):
    emb  = nn.Embedding(V, d_model, device=device, dtype=dtype)
    head = nn.Linear(d_model, V, bias=True, device=device, dtype=dtype)
    return emb, head

def proj_qkv(attn, x):
    x = attn.ln(x)
    qkv = attn.W_qkv(x).view(1, attn.H, 2*attn.d_h + attn.d_v)
    Q, K, Vv = torch.split(qkv, [attn.d_h, attn.d_h, attn.d_v], dim=-1)
    QL = torch.einsum('bhd,hdr->bhr', Q, attn.E_q)
    KL = torch.einsum('bhd,hdr->bhr', K, attn.E_k)
    VL = torch.einsum('bhd,hdr->bhr', Vv, attn.E_v)
    return QL, KL, VL

@torch.no_grad()
def eval_many(attn, head, emb, dev, dty, V=512, GAP=128, n_episodes=200):
    """Mirrors your eval: write key+fill to memory, query at ASK, compute CE+acc."""
    H, r = attn.H, attn.r
    tot_ce, acc = 0.0, 0
    eps = 1e-3
    ASK, FILL = 0, 1
    for _ in range(n_episodes):
        M = torch.zeros(1, H, r, r, device=dev, dtype=dty)
        z = torch.full((1, H, r), eps, device=dev, dtype=dty)
        key = torch.randint(2, V, (1,), device=dev).item()
        seq = [key] + [FILL]*GAP + [ASK]

        for tok in seq[:-1]:
            QL, KL, VL = proj_qkv(attn, emb(torch.tensor([tok], device=dev)))
            sK = F.softplus(KL)
            num_hat = torch.einsum('bhr,bhrr->bhr', sK, M)
            den_hat = (sK * z).sum(-1, keepdim=True).clamp_min(1e-6)
            VhatL  = num_hat / den_hat
            w = sK.unsqueeze(-1) * (VL - VhatL).unsqueeze(-2)
            M.add_(w); z.add_(sK)

        QLq, _, _ = proj_qkv(attn, emb(torch.tensor([ASK], device=dev)))
        sQ = F.softplus(QLq)
        num = torch.einsum('bhr,bhrr->bhr', sQ, M)
        den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
        YL  = num / den
        Y   = torch.einsum('bhr,hrd->bhd', YL, attn.D_v)
        logits = head(Y.reshape(1, -1))
        tgt = torch.tensor([key], device=dev)
        ce = F.cross_entropy(logits, tgt).item()
        tot_ce += ce
        acc += int(logits.argmax(-1).item() == key)
    return tot_ce / n_episodes, acc / n_episodes

def objective(trial, episodes=400, eval_every=50):
    # ---- Search space (no arch changes) - MORE CONSTRAINED
    V = trial.suggest_categorical("V", [256])             # task difficulty (ln(V) ~ 5.55)
    base_lr = trial.suggest_float("lr_main", 1e-5, 5e-4, log=True)
    lr_evdv = trial.suggest_float("lr_evdv", 1e-5, 3e-4, log=True)
    wd      = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    warmup  = trial.suggest_int("warmup_steps", 50, 150)
    clip    = trial.suggest_float("clip_norm", 0.3, 1.0)
    smooth  = trial.suggest_float("label_smoothing", 0.0, 0.2)

    beta_start  = trial.suggest_float("beta_start",  -2.5, -0.6)  # lower => more local
    beta_target = trial.suggest_float("beta_target", -1.6, -0.2)
    tau_start   = trial.suggest_float("tau_start",    0.6,  1.4)
    tau_target  = trial.suggest_float("tau_target",   0.10, 0.6)
    k_top_late  = trial.suggest_int("k_top_final", 1, 3)

    eta_min_ratio = trial.suggest_float("eta_min_ratio", 0.05, 0.5)
    T_max_sched   = trial.suggest_int("T_max", 200, 400)

    # ---- Build attention/state (smaller config for faster tuning) - SMALLER MODEL
    # Ensure d_model is divisible by n_heads and d_value is consistent
    cfg = FusedV2Config(d_model=128, n_heads=2, head_dim=32, d_value=32,
                        r_latent=8, window=32, block_size=8, top_k_blocks=2, bsz=1)
    attn, st, dev, dty = build_fused_v2(cfg)

    # Initialize mixer/router + pseudo-inverse D_v like your script
    torch.manual_seed(0)
    attn.tau_skip = tau_start
    with torch.no_grad():
        attn.beta.data.fill_(beta_start)
        attn.E_q.data.copy_(attn.E_k.data)  # Stable latent init
        for h in range(attn.H):
            Ev = attn.E_v[h].to(torch.float32)
            attn.D_v.data[h].copy_(torch.linalg.pinv(Ev).to(attn.E_v.dtype))

    # Freeze E_v & D_v during warm-up; unfreeze later
    evdv_params = [attn.E_v, attn.D_v]
    for p in evdv_params: p.requires_grad_(False)

    # Identity W_o (when H*d_v == d_model == 256)
    with torch.no_grad():
        assert attn.H * attn.d_v == attn.d_model
        attn.W_o.weight.copy_(torch.eye(attn.d_model, device=dev, dtype=dty))
    attn.W_o.weight.requires_grad_(False)

    # Head + optimizer (smaller head for faster training)
    emb, head = build_vocab_head(cfg.d_model, min(V, 256), dev, dty)  # Cap V at 256 for smaller models

    train_params = list(emb.parameters())
    for n, p in attn.named_parameters():
        if any(n.startswith(k) for k in ["W_qkv", "ln", "E_q", "E_k"]):
            train_params.append(p)
    train_params += list(head.parameters())

    opt = torch.optim.AdamW(train_params, lr=base_lr, weight_decay=wd)
    opt.add_param_group({'params': evdv_params, 'lr': lr_evdv, 'weight_decay': wd})
    train_params += evdv_params

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_sched,
                                                       eta_min=base_lr * eta_min_ratio)

    def run_episode():
        # Same toy episode: write key, then ask
        H, r = attn.H, attn.r
        M = torch.zeros(1, H, r, r, device=dev, dtype=dty)
        z = torch.full((1, H, r), 1e-3, device=dev, dtype=dty)  # Prefill z with epsilon
        ASK, FILL, GAP = 0, 1, 128
        key = torch.randint(2, V, (1,), device=dev).item()
        seq = [key] + [FILL]*GAP + [ASK]
        for tok in seq[:-1]:
            QL, KL, VL = proj_qkv(attn, emb(torch.tensor([tok], device=dev)))
            sK = F.softplus(KL)
            num_hat = torch.einsum('bhr,bhrr->bhr', sK, M)
            den_hat = (sK * z).sum(-1, keepdim=True).clamp_min(1e-6)
            VhatL = num_hat / den_hat
            w = sK.unsqueeze(-1) * (VL - VhatL).unsqueeze(-2)
            M = M + w; z = z + sK
        QLq, _, _ = proj_qkv(attn, emb(torch.tensor([ASK], device=dev)))
        sQ = F.softplus(QLq)
        num = torch.einsum('bhr,bhrr->bhr', sQ, M)
        den = (sQ * z).sum(-1, keepdim=True).clamp_min(1e-6)
        YL  = num / den
        Y   = torch.einsum('bhr,hrd->bhd', YL, attn.D_v)
        logits = head(Y.reshape(1, -1))
        tgt = torch.tensor([key], device=dev)
        return ce_smooth(logits, tgt, smoothing=smooth)

    ema = None
    evdv_unfrozen = False
    best_eval = float("inf")

    for ep in range(1, episodes + 1):
        # Unfreeze E_v/D_v after warm-up
        if not evdv_unfrozen and ep > warmup:
            for p in evdv_params: p.requires_grad_(True)
            evdv_unfrozen = True

        # Cosine schedule beta/tau and k_top
        frac = 0.0 if ep <= warmup else (ep - warmup) / max(1, episodes - warmup)
        cosf = 0.5 * (1 - math.cos(math.pi * frac))
        with torch.no_grad():
            attn.beta.data.fill_(beta_start + cosf * (beta_target - beta_start))
            attn.tau_skip = tau_start + cosf * (tau_target - tau_start)
            attn.k_top = 1 if ep <= warmup else k_top_late

        loss = run_episode()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, clip)
        opt.step(); sched.step()

        L = float(loss.item())
        ema = L if ema is None else 0.98 * ema + 0.02 * L

        # Evaluate and prune periodically
        if ep % eval_every == 0:
            eval_ce, _ = eval_many(attn, head, emb, dev, dty, V=V, n_episodes=200)
            best_eval = min(best_eval, eval_ce)
            trial.report(eval_ce, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=200)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize",
                              pruner=SuccessiveHalvingPruner())
    study.optimize(lambda t: objective(t, args.episodes, args.eval_every),
                   n_trials=args.trials)

    print("Best params:", study.best_params)
    print("Best eval_CE:", study.best_value)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required."
    main()
