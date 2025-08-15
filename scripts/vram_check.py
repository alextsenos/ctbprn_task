"""Estimate and measure VRAM usage of fused_attn_v2 state.
Run: python scripts/vram_check.py
"""
import torch
from fused_attn_v2 import FusedV2Config, build_fused_v2

def vram_estimate_bytes(H, r, w, L, b=2, Bsz=1, R=64, include_s_prev=True, include_block_ids=True):
    kv = 2 * Bsz * H * r * w * L * b                                 # K^L + V^L
    mem = Bsz * H * (r * r + r) * L * b                              # M + z
    s_prev = (Bsz * H * r * r * L * b) if include_s_prev else 0      # S_prev
    block_means = Bsz * H * r * R * L * b                            # R centroids
    block_ids = (Bsz * w * L * 8) if include_block_ids else 0        # int64 ring tags (8 bytes)
    return dict(kv=kv, mem=mem, s_prev=s_prev, block_means=block_means,
                block_ids=block_ids, total=kv + mem + s_prev + block_means + block_ids)

def human_mb(n):
    return n / (1024 * 1024)

if __name__ == "__main__":
    H, r, w, L, Bsz, R = 32, 64, 64, 32, 1, 64
    est = vram_estimate_bytes(H=H, r=r, w=w, L=L, b=2, Bsz=Bsz, R=R)
    print("Static estimate (MB):")
    print({k: round(human_mb(v), 3) for k, v in est.items()})

    if not torch.cuda.is_available():
        print("CUDA not available; skipping runtime measurement.")
        exit()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    cfg = FusedV2Config(window=w, block_size=8, bsz=Bsz)
    attn, st, dev, dty = build_fused_v2(cfg)

    def allocated_mb():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(dev) / (1024 * 1024)

    before = allocated_mb()
    # one step to materialize all buffers
    x = torch.randn(cfg.bsz, cfg.d_model, device=dev, dtype=dty)
    _ = attn(x, st)
    after = allocated_mb()
    peak = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    print(f"delta_allocated ~ {after - before:.3f} MB")
    print(f"peak_allocated  ~ {peak:.3f} MB")
