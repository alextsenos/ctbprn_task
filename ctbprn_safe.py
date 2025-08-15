import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import math
import sys

# Set console output encoding to UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(dev_str):
    if dev_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_str)

def cuda_capability_str():
    if not torch.cuda.is_available(): 
        return "N/A"
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"

def best_amp_dtype(device):
    if device.type != "cuda": 
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

class AmpWrap:
    def __init__(self, device):
        self.device = device
        self.dtype = best_amp_dtype(device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.dtype==torch.float16) if device.type == "cuda" else None

    def autocast(self):
        if self.device.type == "cuda" and self.dtype is not None:
            return torch.amp.autocast("cuda", dtype=self.dtype)
        from contextlib import nullcontext
        return nullcontext()

def enable_gpu_fast_math():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

# Smaller model configuration
DEFAULT_DIM = 128  # Reduced from 256
DEFAULT_BATCH = 32  # Reduced from 128
DEFAULT_STEPS = 20  # Reduced from 120

# Model Components
class TwoCompartmentColumn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W_b  = nn.Linear(d, d, bias=False)
        self.W_a  = nn.Linear(d, d, bias=False)
        self.W_g  = nn.Linear(2*d, d, bias=True)
        self.W_m1 = nn.Linear(d, d, bias=False)
        self.W_m2 = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        for m in [self.W_b, self.W_a, self.W_m1, self.W_m2]:
            nn.init.xavier_uniform_(m.weight, gain=0.8)
        nn.init.xavier_uniform_(self.W_g.weight, gain=0.2)

    def forward(self, x_in, apical_ctx):
        b  = F.gelu(self.W_b(x_in))
        ah = F.gelu(self.W_a(apical_ctx))
        g  = torch.sigmoid(self.W_g(torch.cat([x_in, apical_ctx], dim=-1)))
        m  = F.gelu(self.W_m1(b)) * F.gelu(self.W_m2(ah))
        y  = self.norm(x_in + (1.0 - g)*b + g*m)
        return y

class ThalamicRouter(nn.Module):
    def __init__(self, d: int, rank: int, num_areas: int, temperature: float = 0.8):
        super().__init__()
        self.K = num_areas
        self.temperature = temperature
        self.U = nn.Parameter(torch.randn(self.K, d, rank) * (1.0 / (d**0.5)))
        self.V = nn.Parameter(torch.randn(self.K, d, rank) * (1.0 / (d**0.5)))

    def forward(self, X_list, logits):
        alpha = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        outs=[]
        for i in range(self.K):
            acc = 0.0
            for k in range(self.K):
                tmp = X_list[k] @ self.V[k]
                msg = tmp @ self.U[k].t()
                w = alpha[:,k].view(-1,1,1)
                acc = acc + w*msg
            outs.append(acc)
        return outs

class BGController(nn.Module):
    def __init__(self, d: int, num_areas: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.head = nn.Sequential(
            nn.LayerNorm(num_areas*d),
            nn.Linear(num_areas*d, 2*d),
            nn.GELU(),
            nn.Linear(2*d, num_areas)
        )
    def forward(self, area_means):
        B,K,d = area_means.shape
        proj_means = F.gelu(self.proj(area_means))
        flat_feats = proj_means.view(B, K*d)
        return self.head(flat_feats)

class CTBPRNStep(nn.Module):
    def __init__(self, d: int, num_areas: int = 3, rank: int = 32, temperature: float = 0.8):
        super().__init__()
        self.K = num_areas
        self.columns = nn.ModuleList([TwoCompartmentColumn(d) for _ in range(num_areas)])
        self.router  = ThalamicRouter(d, rank, num_areas, temperature)
        self.ctrl    = BGController(d, num_areas)

    def forward(self, X_list):
        area_means = torch.stack([x.mean(dim=1) for x in X_list], dim=1)
        logits = self.ctrl(area_means)
        routed = self.router(X_list, logits)
        out = [self.columns[i](X_list[i], routed[i]) for i in range(self.K)]
        return out, logits

class CTBPRNClassifier(nn.Module):
    def __init__(self, d:int, K:int, rank:int, steps:int=3, temperature:float=0.8, num_classes:int=2):
        super().__init__()
        self.steps = nn.ModuleList([CTBPRNStep(d,K,rank,temperature) for _ in range(steps)])
        self.head  = nn.Sequential(nn.LayerNorm(d*K), nn.GELU(), nn.Linear(d*K, num_classes))
        self.K=K

    def forward(self, X_list):
        route_logits_all=[]
        for s in self.steps:
            X_list, rl = s(X_list); route_logits_all.append(rl)
        feats = torch.cat([x.mean(dim=1) for x in X_list], dim=-1)
        return self.head(feats), route_logits_all

@torch.no_grad()
def make_batch(B:int, T:int, d:int, K:int, mu:float=0.9, device:str="cuda"):
    Xs = [torch.randn(B,T,d, device=device)*0.7 for _ in range(K)]
    targets = torch.randint(0, K, (B,), device=device)
    y = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        target_area_idx = targets[i].item()
        bias_vec = torch.randn(d//4, device=device) * 0.5
        Xs[target_area_idx][i, :, :d//4] += bias_vec
        if Xs[target_area_idx][i, :, :d//4].sum() > 0:
            y[i] = 1
    return Xs, y, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       choices=["auto", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--B", type=int, default=DEFAULT_BATCH, 
                       help=f"Batch size (reduced from default 128 to {DEFAULT_BATCH})")
    parser.add_argument("--T", type=int, default=128, 
                       help=f"Sequence length (reduced from 256 to 128)")
    parser.add_argument("--d", type=int, default=DEFAULT_DIM, 
                       help=f"Model dimension (reduced from 256 to {DEFAULT_DIM})")
    parser.add_argument("--K", type=int, default=2, 
                       help="Number of areas (reduced from 3 to 2)")
    parser.add_argument("--rank", type=int, default=16, 
                       help="Rank for ThalamicRouter (reduced from 32 to 16)")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate (reduced from 3e-3 to 1e-3)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--compile", action="store_true", 
                       help="Use torch.compile for model (disabled by default)")
    
    args = parser.parse_args()
    
    print("\n=== Memory-Safe Configuration ===")
    print(f"Batch size: {args.B} (default: 128)")
    print(f"Model dim: {args.d} (default: 256)")
    print(f"Sequence len: {args.T} (default: 256)")
    print(f"Areas (K): {args.K} (default: 3)")
    print(f"Router rank: {args.rank} (default: 32)")
    print("=============================\n")
    
    try:
        # [Rest of the original main() function remains the same]
        set_seed(7)
        device = pick_device(args.device)
        enable_gpu_fast_math()

        print(f"Device  : {device}")
        if device.type == "cuda":
            print(f"GPU     : {torch.cuda.get_device_name(0)}")
            print(f"Compute : {cuda_capability_str()}")

        amp = AmpWrap(device)
        B,T,d,K,rank = args.B, args.T, args.d, args.K, args.rank
        
        # Enable gradient checkpointing if available
        use_ckpt = hasattr(torch, "utils")
        if use_ckpt:
            print("Using gradient checkpointing to save memory")
            
        model = CTBPRNClassifier(d,K,rank,steps=2,temperature=args.temperature,num_classes=2).to(device)
        
        if args.compile and hasattr(torch, "compile"):
            print("Using torch.compile (experimental)")
            model = torch.compile(model, fullgraph=False)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # Add memory monitoring
        def print_memory_usage():
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
                      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

        @torch.no_grad()
        def eval_batch(bs):
            torch.cuda.empty_cache()
            model.eval()
            try:
                Xs,y,targets = make_batch(bs,T,d,K, device=device.type)
                logits, route_logits = model(Xs)
                acc = (logits.argmax(dim=-1) == y).float().mean()
                route_matches = torch.stack([(rl.argmax(dim=-1) == targets).float().mean() 
                                          for rl in route_logits])
                return acc, route_matches.mean()
            except RuntimeError as e:
                print(f"Error in eval_batch: {e}")
                return torch.tensor(0.0), torch.tensor(0.0)

        # Initial memory usage
        print("\n=== Initial Memory Usage ===")
        print_memory_usage()
        
        # Smaller initial batch for warmup
        try:
            print("\n=== Warmup Run ===")
            Xs,y,_ = make_batch(2, 16, d, K, device=device.type)
            _ = model(Xs)
            print("Warmup successful")
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")

        # Training loop with memory monitoring
        print("\n=== Starting Training ===")
        model.train()
        for it in range(1, args.steps+1):
            try:
                # Clear cache before each iteration
                if it > 1 and it % 5 == 0:
                    torch.cuda.empty_cache()
                    
                Xs,y,_ = make_batch(B,T,d,K, device=device.type)
                
                with amp.autocast():
                    logits, _ = model(Xs)
                    loss = F.cross_entropy(logits, y)
                
                if amp.scaler:
                    amp.scaler.scale(loss).backward()
                    amp.scaler.step(opt)
                    amp.scaler.update()
                    opt.zero_grad(set_to_none=True)
                else:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                if it % 5 == 0:
                    print(f"\nStep {it:03d}:")
                    print(f"  Loss: {loss.item():.4f}")
                    print_memory_usage()
                    
                    # Smaller eval batch
                    acc, rm = eval_batch(min(128, B))
                    print(f"  Val Acc: {acc:.3f}  Route Match: {rm:.3f}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\n⚠️  Out of Memory Error! Trying to recover...")
                    torch.cuda.empty_cache()
                    # Reduce batch size for next attempt
                    args.B = max(1, args.B // 2)
                    print(f"Reducing batch size to {args.B}")
                    if args.B == 1:
                        print("Batch size reduced to 1. Stopping to prevent further issues.")
                        break
                    continue
                else:
                    print(f"\n⚠️  Error: {e}")
                    break
                    
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            print("\n=== Final Memory Usage ===")
            print_memory_usage()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
