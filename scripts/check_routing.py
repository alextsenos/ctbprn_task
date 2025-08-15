import torch
from fused_attn_v2 import FusedV2Config, build_fused_v2

assert torch.cuda.is_available(), "CUDA required."

# Small, fast config for routing sanity
cfg = FusedV2Config(
    d_model=256, n_heads=4, head_dim=32, d_value=32,
    r_latent=8, window=32, block_size=4, top_k_blocks=2, bsz=1,
)

attn, st, dev, dty = build_fused_v2(cfg)
torch.manual_seed(0)

# ---- Fill exactly 6 full blocks (IDs 0..5) so cur_block_id becomes 6
with torch.no_grad():
    for _ in range(cfg.block_size * 6):  # 6 full blocks
        _ = attn(torch.randn(1, cfg.d_model, device=dev, dtype=dty), st)

print("cur_block_id after 6 full blocks:", st.cur_block_id)  # expect 6
# At this point, ring has blocks 0..5; block 6 has *no tokens yet*.

# ---- Start block 6 by appending ONE MORE token, so block_id==6 exists in the ring
with torch.no_grad():
    _ = attn(torch.randn(1, cfg.d_model, device=dev, dtype=dty), st)

# ---- Overwrite stored block means with controlled prototypes for blocks 0..5
B, H, r = st.bsz, st.H, st.r
means = []
for i in range(st.cur_block_id):  # block means stored for past blocks 0..5
    v = torch.zeros(B, H, r, device=dev, dtype=dty)
    v[..., i % r] = 1.0
    means.append(v)
st.block_means = means  # list of [B,H,r] tensors

# ---- Craft a Q^L that aligns with blocks 1 and 4 (top-2)
QL = torch.zeros(B, H, r, device=dev, dtype=dty)
QL[..., 1] = 2.0
QL[..., 4] = 1.5

# ---- Compute routed mask: should include current block (6) + top-2 {1,4}
mask = st.routed_mask(QL, k_top=2)  # [B,1,w] boolean

# Gather which block IDs are actually selected among ring slots
sel_ids = st.block_ids[0][mask[0, 0]].unique().tolist()
sel_ids.sort()
print("selected block IDs (should contain current=6 plus top-2={1,4}):", sel_ids)

# Optional safety checks
assert 6 in sel_ids, "Current block (6) missing—did you append the extra token?"
assert 1 in sel_ids and 4 in sel_ids, "Top-2 routed blocks not present as expected."
print("Routing sanity: PASS")
