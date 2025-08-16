import torch
from fused_attn_v2.fused_v2 import FusedV2HRMModel, HRMConfig, deep_supervision_step, autocast_dtype

# Configuration
VOCAB_SIZE = 32000  # Replace with your actual vocabulary size
BATCH_SIZE = 4
SEQ_LEN = 4096

def get_dummy_batch(device, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE):
    """Generate a dummy batch for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size,), device=device)  # For per_token_head=False
    attn_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    return input_ids, labels, attn_mask

def main():
    # Set up device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize configuration
    cfg = HRMConfig(
        vocab_size=VOCAB_SIZE,
        max_len=SEQ_LEN,
        d_model=512,  # Reduced for testing
        n_heads=8,
        d_ff=2048,
        n_layers_L=4,  # Fewer layers for testing
        n_layers_H=2,
        per_token_head=False,
        use_act=True,
        Mmax=4,
        epsilon=0.1,
        window=256,
        dilation=4,
        K_latent=32,
        M_mem=32
    )
    
    model = FusedV2HRMModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training loop
    num_batches = 10  # For testing
    for batch_idx in range(num_batches):
        # Get dummy batch
        input_ids, labels, attn_mask = get_dummy_batch(device)
        
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
                epsilon=cfg.epsilon
            )
        
        # Print training stats
        print(f"Batch {batch_idx + 1}/{num_batches}:")
        print(f"  Sequence Loss: {stats['seq_loss']:.4f}")
        print(f"  Q-Loss: {stats['q_loss']:.4f}")
        print(f"  Segments: {stats['segments']}")
        print(f"  Halted: {stats['halted']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
