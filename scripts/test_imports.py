print("Testing imports...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    from fused_attn_v2 import FusedV2Config, build_fused_v2
    print("Successfully imported fused_attn_v2")
    
    # Test a simple model
    cfg = FusedV2Config(
        d_model=64,
        n_heads=8,
        head_dim=8,
        d_value=8,
        r_latent=16,
        window=64,
        block_size=8,
        top_k_blocks=2,
        tau_skip=-1.0,
        bsz=1
    )
    print("Creating model...")
    attn, st, dev, dty = build_fused_v2(cfg)
    print(f"Model created successfully on device: {dev}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
