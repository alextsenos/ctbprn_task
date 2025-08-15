import joblib
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_study(study_path):
    # Load the study
    study = joblib.load(study_path)
    
    # Get all trials
    trials = study.trials
    
    # Extract trial details
    data = []
    for trial in trials:
        if trial.value is not None:  # Only include completed trials
            data.append({
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Extract parameters into separate columns
    for param in study.best_params.keys():
        df[param] = df['params'].apply(lambda x: x.get(param, None))
    
    # Sort by trial number
    df = df.sort_values('trial')
    
    # Print summary
    print("\n=== Study Analysis ===")
    print(f"Number of trials: {len(df)}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Plot validation accuracy vs trials
    plt.figure(figsize=(10, 5))
    plt.plot(df['trial'], df['value'], 'o-')
    plt.xlabel('Trial')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Trial')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('optuna_plots', exist_ok=True)
    plot_path = os.path.join('optuna_plots', 'validation_accuracy.png')
    plt.savefig(plot_path)
    print(f"\nSaved validation accuracy plot to: {plot_path}")
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for name, score in importance.items():
            print(f"  {name}: {score:.4f}")
    except Exception as e:
        print(f"\nCould not calculate parameter importance: {e}")
    
    return df

if __name__ == "__main__":
    study_path = "optuna_studies/ctbprn_study.pkl"
    if os.path.exists(study_path):
        print(f"Analyzing study: {study_path}")
        df = analyze_study(study_path)
    else:
        print(f"Study not found at: {study_path}")
        print("Please run the tuning script first.")
