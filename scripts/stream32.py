import torch, torch.nn as nn, torch.nn.functional as F
from fused_attn_v2 import FusedV2Config, build_fused_v2
cfg=FusedV2Config(window=64, block_size=8, bsz=1)
attn, st, dev, dty = build_fused_v2(cfg)
emb, head = nn.Embedding(32000,cfg.d_model).to(dev,dty), nn.Linear(cfg.d_model,32000,bias=False).to(dev,dty)
opt = torch.optim.AdamW([*emb.parameters(), *head.parameters(), *attn.parameters()], lr=1e-3)

torch.manual_seed(0)
loss_sum=0.0
for _ in range(32):
    tok = torch.randint(0,32000,(1,),device=dev)
    y = attn(emb(tok), st)
    logits = head(y)
    loss = F.cross_entropy(logits, torch.randint(0,32000,(1,),device=dev))
    # state must be grad-free between steps:
    assert not st.M.requires_grad and not st.z.requires_grad
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    loss_sum += loss.detach().item()
print("OK — 32 steps, total CE:", round(loss_sum,3))