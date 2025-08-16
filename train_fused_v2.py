import torch
from fused_attn_v2.fused_v2 import FusedV2HRMModel, HRMConfig, deep_supervision_step, autocast_dtype

# Configuration
VOCAB_SIZE = 32000  # Replace with your actual vocabulary size
BATCH_SIZE = 2  # Reduced from 4 to 2 to save memory
SEQ_LEN = 2048  # Reduced sequence length from 4096 to 2048

def get_dummy_batch(device, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, per_token_head=False):
    """Generate a dummy batch with a learnable pattern."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    if per_token_head:
        # For per-token prediction (e.g., next-token prediction)
        labels = input_ids.clone()
    else:
        # For sequence-level prediction: sum of tokens mod vocab_size
        labels = input_ids.sum(dim=1) % vocab_size
    attn_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    return input_ids, labels, attn_mask

def main():
    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize configuration with memory-efficient settings
    cfg = HRMConfig(
        vocab_size=VOCAB_SIZE,
        max_len=SEQ_LEN,
        d_model=256,  # Further reduced from 384
        n_heads=4,    # Reduced from 6
        d_ff=1024,    # Reduced from 1536
        n_layers_L=2,  # Reduced from 3
        n_layers_H=1,  # Keep at 1
        per_token_head=False,
        use_act=True,
        Mmax=4,
        epsilon=0.1,
        window=64,    # Reduced from 128
        dilation=4,
        K_latent=16,  # Reduced from 24
        M_mem=16,     # Reduced from 24
        grad_ckpt=False,  # Disable gradient checkpointing temporarily
        use_autocast=True  # Keep mixed precision training
    )
    
    model = FusedV2HRMModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training loop
    num_batches = 1000  # Increased for better learning
    global_step = 0
    
    print("Starting training with ACT warmup...")
    print(f"ACT will be enabled after {200} steps")
    print(f"Minimum segments: {2}, Q-loss weight: {0.25}, Halt margin: {0.1}")
    print("-" * 50)
    
    for batch_idx in range(num_batches):
        # Get dummy batch - now with learnable pattern
        input_ids, labels, attn_mask = get_dummy_batch(device, per_token_head=cfg.per_token_head)
        
        # Initialize state
        B, L = input_ids.shape
        state = model.core.init_state(B, L, device=input_ids.device)
        
        # Forward pass with autocast
        with torch.autocast("cuda", enabled=cfg.use_autocast, dtype=autocast_dtype()):
            stats = deep_supervision_step(
                model=model.core,
                input_ids=input_ids,
                targets=labels,
                attention_mask=attn_mask,
                state=state,
                optimizer=optimizer,
                use_act=cfg.use_act,
                Mmax=cfg.Mmax,
                epsilon=cfg.epsilon,
                step=global_step,
                act_warmup_steps=200,
                min_segments=2,
                q_weight=0.25,
                halt_margin=0.1
            )
        
        # Print training stats
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"Step {global_step + 1}/{num_batches}:")
            print(f"  Sequence Loss: {stats['seq_loss']:.4f}")
            print(f"  Q-Loss: {stats['q_loss']:.4f}")
            print(f"  Segments: {stats['segments']}")
            print(f"  Halted: {stats['halted']}")
            if stats['last_q'] is not None:
                q_probs = torch.sigmoid(stats['last_q'])
                print(f"  Q-Values (halt, cont): {q_probs[0].tolist()}")
            print("-" * 50)
        
        global_step += 1

if __name__ == "__main__":
    main()
