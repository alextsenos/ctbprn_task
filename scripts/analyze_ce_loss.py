import joblib
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_ce_loss(study_path):
    # Load the study
    study = joblib.load(study_path)
    
    # Extract trial data
    trials = study.trials
    data = []
    
    for trial in trials:
        if trial.value is not None:  # Only include completed trials
            ce_loss = trial.user_attrs.get('ce_loss', float('nan'))
            data.append({
                'trial': trial.number,
                'val_accuracy': trial.value,
                'ce_loss': ce_loss,
                'lr': trial.params.get('lr'),
                'd': trial.params.get('d'),
                'rank': trial.params.get('rank'),
                'temperature': trial.params.get('temperature'),
                'batch_size': trial.params.get('batch_size'),
                'seq_len': trial.params.get('seq_len')
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Print summary
    print("\n=== Cross-Entropy Loss Analysis ===")
    print(f"Number of trials: {len(df)}")
    print(f"Average CE Loss: {df['ce_loss'].mean():.4f} ± {df['ce_loss'].std():.4f}")
    print(f"Best CE Loss: {df['ce_loss'].min():.4f}")
    
    # Find best trial by CE loss
    best_ce_idx = df['ce_loss'].idxmin()
    print("\nBest trial by CE Loss:")
    print(df.loc[best_ce_idx])
    
    # Plot CE Loss vs Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ce_loss'], df['val_accuracy'], alpha=0.7)
    plt.xlabel('Cross-Entropy Loss (lower is better)')
    plt.ylabel('Validation Accuracy (higher is better)')
    plt.title('CE Loss vs Validation Accuracy')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('optuna_plots', exist_ok=True)
    plot_path = os.path.join('optuna_plots', 'ce_loss_vs_accuracy.png')
    plt.savefig(plot_path)
    print(f"\nSaved CE Loss vs Accuracy plot to: {plot_path}")
    
    # Plot CE Loss vs Learning Rate
    plt.figure(figsize=(10, 6))
    plt.scatter(df['lr'], df['ce_loss'], alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('CE Loss vs Learning Rate')
    plt.grid(True)
    
    plot_path = os.path.join('optuna_plots', 'ce_loss_vs_lr.png')
    plt.savefig(plot_path)
    print(f"Saved CE Loss vs Learning Rate plot to: {plot_path}")
    
    return df

if __name__ == "__main__":
    study_path = "optuna_studies/ctbprn_study.pkl"
    if os.path.exists(study_path):
        print(f"Analyzing study: {study_path}")
        df = analyze_ce_loss(study_path)
        
        # Save detailed results to CSV
        csv_path = "optuna_results_detailed.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detailed results to: {csv_path}")
    else:
        print(f"Study not found at: {study_path}")
        print("Please run the tuning script first.")
