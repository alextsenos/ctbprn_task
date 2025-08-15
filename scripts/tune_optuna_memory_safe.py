import optuna
from optuna.samplers import TPESampler
import torch
import gc
import os
from pathlib import Path
import sys

# Add parent directory to path to import ctbprn_safe
sys.path.append(str(Path(__file__).parent.parent))
from ctbprn_safe import CTBPRNClassifier, make_batch, set_seed

def objective(trial):
    # Suggest hyperparameters with conservative ranges
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'd': trial.suggest_categorical('d', [32, 64, 96]),
        'rank': trial.suggest_categorical('rank', [4, 8, 16]),
        'temperature': trial.suggest_float('temperature', 0.5, 2.0),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'seq_len': trial.suggest_categorical('seq_len', [32, 64, 128])
    }
    
    # Fixed parameters to keep memory usage low
    K = 2  # Number of areas
    steps = 2  # Reduced number of CTBPRN steps
    num_epochs = 5  # Reduced number of epochs
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    
    try:
        # Create model with suggested parameters
        model = CTBPRNClassifier(
            d=params['d'],
            K=K,
            rank=params['rank'],
            steps=steps,
            temperature=params['temperature'],
            num_classes=2
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
        
        # Training loop with memory management
        best_val_acc = 0
        epoch_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            
            # Generate small batch for this trial
            Xs, y, _ = make_batch(
                B=params['batch_size'],
                T=params['seq_len'],
                d=params['d'],
                K=K,
                device=device.type
            )
            
            # Forward pass
            logits, _ = model(Xs)
            loss = torch.nn.functional.cross_entropy(logits, y)
            
            # Store the loss for this epoch
            epoch_losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_Xs, val_y, _ = make_batch(
                    B=min(32, params['batch_size'] * 2),  # Small validation set
                    T=params['seq_len'],
                    d=params['d'],
                    K=K,
                    device=device.type
                )
                val_logits, _ = model(val_Xs)
                val_acc = (val_logits.argmax(dim=-1) == val_y).float().mean().item()
                
                # Update best validation accuracy
                best_val_acc = max(best_val_acc, val_acc)
            
            # Clear memory
            del Xs, y, logits, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            # Early stopping if performance is too poor
            if epoch > 1 and best_val_acc < 0.4:
                break
        
        # Calculate average CE loss over all epochs
        avg_ce_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Store CE loss in trial user attributes for analysis
        trial.set_user_attr('ce_loss', avg_ce_loss)
        
        return best_val_acc
        
    except RuntimeError as e:
        # Handle out of memory errors
        if 'out of memory' in str(e).lower():
            print(f"OOM with params: {params}")
            return 0.0
        raise

def optimize_hyperparameters(n_trials=20, study_name="ctbprn_optuna_study"):
    # Set up study with memory-efficient settings
    sampler = TPESampler(n_startup_trials=5, seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    
    # Print results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (val acc): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study

if __name__ == "__main__":
    print("Starting memory-efficient hyperparameter optimization...")
    print("This will run with small batch sizes and model dimensions to prevent OOM.")
    print("Press Ctrl+C to stop early and keep the best results.\n")
    
    try:
        # Run with just 10 trials to start
        study = optimize_hyperparameters(n_trials=10)
        
        # Save study
        import joblib
        os.makedirs("optuna_studies", exist_ok=True)
        joblib.dump(study, "optuna_studies/ctbprn_study.pkl")
        print("\nStudy saved to optuna_studies/ctbprn_study.pkl")
        
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
    finally:
        # Ensure all CUDA memory is released
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Optimization completed.")
