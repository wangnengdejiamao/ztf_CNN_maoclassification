"""
Training Curve Visualization

This module generates comparative plots of training metrics including
loss and accuracy curves for different models.

Usage:
    python plot_curve.py

Output:
    curve.png - Comparative training curves

Author: [Your Name]
Date: 2023
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def load_training_logs(model_names):
    """
    Load training logs for specified models.
    
    Args:
        model_names: List of model names
        
    Returns:
        Dictionary mapping model names to log DataFrames
    """
    logs = {}
    for name in model_names:
        try:
            log_df = pd.read_csv(f'{name}.log')
            logs[name] = log_df
        except FileNotFoundError:
            print(f"Warning: Log file not found for {name}")
    return logs


def plot_training_curves(logs, output_file='curve.png'):
    """
    Generate comparative training curve plots.
    
    Args:
        logs: Dictionary of model logs
        output_file: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training loss
    ax = axes[0, 0]
    for name, log_df in logs.items():
        ax.plot(log_df['loss'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot validation loss
    ax = axes[0, 1]
    for name, log_df in logs.items():
        ax.plot(log_df['test_loss'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot training accuracy
    ax = axes[1, 0]
    for name, log_df in logs.items():
        ax.plot(log_df['acc'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax = axes[1, 1]
    for name, log_df in logs.items():
        ax.plot(log_df['test_acc'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_file}")


def main():
    """
    Main visualization routine.
    """
    # Models to include in comparison
    model_names = ['ghostnet', 'mobilenetv2']
    
    print("Loading training logs...")
    logs = load_training_logs(model_names)
    
    if not logs:
        print("No training logs found!")
        return
    
    print(f"Found logs for: {list(logs.keys())}")
    
    print("Generating plots...")
    plot_training_curves(logs)
    
    print("Done!")


if __name__ == '__main__':
    main()
