#!/usr/bin/env python3
"""
Script to analyze and visualize training logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import numpy as np


def load_log_data(log_path):
    """Load training log data from CSV or JSON file."""
    log_path = Path(log_path)
    
    if log_path.suffix == '.csv':
        # Load CSV data
        df = pd.read_csv(log_path)
        return df
    elif log_path.suffix == '.json':
        # Load JSON data and convert to DataFrame
        with open(log_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['metrics'])
        return df
    else:
        raise ValueError(f"Unsupported file format: {log_path.suffix}")


def filter_data_by_epochs(df, max_epochs):
    """Filter dataframe to only include data up to max_epochs."""
    if max_epochs is None:
        return df
    
    # Filter data where epoch <= max_epochs
    filtered_df = df[df['epoch'] <= max_epochs].copy()
    
    if len(filtered_df) == 0:
        print(f"Warning: No data found for epochs <= {max_epochs}")
        return df
    
    print(f"Filtered data: {len(df)} -> {len(filtered_df)} batches (epochs <= {max_epochs})")
    return filtered_df


def get_epoch_end_data(df):
    """Extract only the last batch data for each epoch (end-of-epoch data)."""
    # Check if we have epoch summary entries (batch = -1) with statistics
    if 'train_loss_std' in df.columns and (df['batch'] == -1).any():
        # Prefer epoch summary entries if available
        epoch_summary_df = df[df['batch'] == -1].copy()
        # Also get regular epoch-end data for epochs without summary
        regular_epoch_end = df[df['batch'] != -1].groupby('epoch').last().reset_index()
        
        # Combine them, prioritizing summary entries
        combined_epochs = set(epoch_summary_df['epoch']).union(set(regular_epoch_end['epoch']))
        result_rows = []
        
        for epoch in sorted(combined_epochs):
            summary_row = epoch_summary_df[epoch_summary_df['epoch'] == epoch]
            if not summary_row.empty:
                result_rows.append(summary_row.iloc[0])
            else:
                regular_row = regular_epoch_end[regular_epoch_end['epoch'] == epoch]
                if not regular_row.empty:
                    result_rows.append(regular_row.iloc[0])
        
        if result_rows:
            epoch_end_df = pd.DataFrame(result_rows)
            return epoch_end_df
    
    # Fallback to original behavior
    epoch_end_df = df.groupby('epoch').last().reset_index()
    return epoch_end_df


def plot_training_curves(df, output_dir=None, experiment_name="training", max_epochs=None):
    """Create training curve plots in 2x2 layout."""
    # Filter data if max_epochs is specified
    df = filter_data_by_epochs(df, max_epochs)
    
    # Extract end-of-epoch data for cleaner plots
    epoch_df = get_epoch_end_data(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    title = f'{experiment_name} - Training Progress'
    if max_epochs is not None:
        title += f' (up to epoch {max_epochs})'
    fig.suptitle(title, fontsize=16)
    
    # Check if we have standard deviation data
    has_std_data = 'train_loss_std' in epoch_df.columns and not epoch_df['train_loss_std'].isna().all()
    
    # Top Left: Training Loss vs Wall Time (using epoch end data)
    ax1 = axes[0, 0]
    if has_std_data:
        ax1.errorbar(epoch_df['wall_time_minutes'], epoch_df['train_loss'], 
                    yerr=epoch_df['train_loss_std'], color='blue', alpha=0.8, 
                    linewidth=2, marker='o', markersize=3, capsize=3, capthick=1, elinewidth=1)
    else:
        ax1.plot(epoch_df['wall_time_minutes'], epoch_df['train_loss'], 'b-', alpha=0.8, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Wall Time (minutes)')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs Wall Time (Per Epoch)')
    ax1.grid(True, alpha=0.3)
    
    # Top Right: Training Loss vs Epochs
    ax2 = axes[0, 1]
    if has_std_data:
        ax2.errorbar(epoch_df['epoch'], epoch_df['train_loss'], 
                    yerr=epoch_df['train_loss_std'], color='blue', alpha=0.8, 
                    linewidth=2, marker='o', markersize=3, capsize=3, capthick=1, elinewidth=1)
    else:
        ax2.plot(epoch_df['epoch'], epoch_df['train_loss'], 'b-', alpha=0.8, linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss vs Epochs')
    ax2.grid(True, alpha=0.3)
    
    # Bottom Left: Validation Loss vs Wall Time
    ax3 = axes[1, 0]
    ax3.plot(epoch_df['wall_time_minutes'], epoch_df['val_loss'], '#ff7f0e', alpha=0.8, linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('Wall Time (minutes)')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Validation Loss vs Wall Time (Per Epoch)')
    ax3.grid(True, alpha=0.3)
    
    # Bottom Right: Validation Accuracy vs Wall Time
    ax4 = axes[1, 1]
    ax4.plot(epoch_df['wall_time_minutes'], epoch_df['val_accuracy'], 'g-', alpha=0.8, linewidth=2, marker='^', markersize=3)
    ax4.set_xlabel('Wall Time (minutes)')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_title('Validation Accuracy vs Wall Time (Per Epoch)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        output_path = Path(output_dir) / f"{experiment_name}_training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {output_path}")
    
    plt.show()


def print_summary_stats(df, experiment_name="training"):
    """Print summary statistics from the training log."""
    print(f"\n=== {experiment_name} Summary Statistics ===")
    print(f"Total training time: {df['wall_time_minutes'].max():.2f} minutes")
    print(f"Total batches: {len(df)}")
    
    # Check if we have epoch 0 data and calculate improvement
    epoch_0_data = df[df['epoch'] == 0]
    final_accuracy = df['val_accuracy'].iloc[-1]
    best_accuracy = df['val_accuracy'].max()
    
    if not epoch_0_data.empty:
        initial_acc = epoch_0_data['val_accuracy'].iloc[0]
        print(f"Initial validation accuracy (Epoch 0): {initial_acc:.2f}%")
        print(f"Final validation accuracy: {final_accuracy:.2f}% (Δ: +{final_accuracy-initial_acc:.2f}%)")
        print(f"Best validation accuracy: {best_accuracy:.2f}% (Δ: +{best_accuracy-initial_acc:.2f}%)")
    else:
        print(f"Final validation accuracy: {final_accuracy:.2f}%")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    print(f"Final training loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final validation loss: {df['val_loss'].iloc[-1]:.4f}")
    
    # Find epoch/batch where best accuracy occurred
    best_idx = df['val_accuracy'].idxmax()
    best_time = df['wall_time_minutes'].iloc[best_idx]
    print(f"Best accuracy achieved at: {best_time:.2f} minutes")
    
    if 'samples_per_second' in df.columns:
        avg_throughput = df['samples_per_second'].mean()
        print(f"Average training throughput: {avg_throughput:.1f} samples/second")


def compare_experiments(log_paths, labels=None, output_dir=None, max_epochs=None):
    """Compare multiple experiments using 2x2 layout."""
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(log_paths))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    title = 'Experiment Comparison'
    if max_epochs is not None:
        title += f' (up to epoch {max_epochs})'
    fig.suptitle(title, fontsize=16)
    
    # Define better colors for comparison: blue and orange with transparency
    if len(log_paths) == 2:
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        markers = ['o', 's']  # Circle and square
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, len(log_paths)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(log_paths)]
    
    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        df = load_log_data(log_path)
        # Filter data if max_epochs is specified
        df = filter_data_by_epochs(df, max_epochs)
        
        # Extract end-of-epoch data for cleaner comparison
        epoch_df = get_epoch_end_data(df)
        
        color = colors[i]
        marker = markers[i] if len(log_paths) <= len(markers) else 'o'
        
        # Top Left: Training Loss vs Wall Time (using epoch end data)
        has_std_data = 'train_loss_std' in epoch_df.columns and not epoch_df['train_loss_std'].isna().all()
        
        if has_std_data:
            # Plot with error bars showing standard deviation
            axes[0, 0].errorbar(epoch_df['wall_time_minutes'], epoch_df['train_loss'], 
                               yerr=epoch_df['train_loss_std'], 
                               label=label, alpha=0.8, linewidth=2, color=color,
                               marker=marker, markersize=3, markevery=max(1, len(epoch_df)//20),
                               capsize=3, capthick=1, elinewidth=1)
        else:
            axes[0, 0].plot(epoch_df['wall_time_minutes'], epoch_df['train_loss'], 
                           label=label, alpha=0.8, linewidth=2, color=color, 
                           marker=marker, markersize=3, markevery=max(1, len(epoch_df)//20))
        
        # Top Right: Training Loss vs Epochs
        if has_std_data:
            # Plot with error bars showing standard deviation
            axes[0, 1].errorbar(epoch_df['epoch'], epoch_df['train_loss'], 
                               yerr=epoch_df['train_loss_std'],
                               label=label, alpha=0.8, linewidth=2, color=color,
                               marker=marker, markersize=3, markevery=max(1, len(epoch_df)//20),
                               capsize=3, capthick=1, elinewidth=1)
        else:
            axes[0, 1].plot(epoch_df['epoch'], epoch_df['train_loss'], 
                           label=label, alpha=0.8, linewidth=2, color=color,
                           marker=marker, markersize=3, markevery=max(1, len(epoch_df)//20))
        
        # Bottom Left: Validation Loss vs Wall Time
        axes[1, 0].plot(epoch_df['wall_time_minutes'], epoch_df['val_loss'], 
                       label=label, alpha=0.8, linewidth=2, color=color,
                       marker=marker, markersize=3, markevery=max(1, len(epoch_df)//20))
        
        # Bottom Right: Validation Accuracy vs Wall Time
        axes[1, 1].plot(epoch_df['wall_time_minutes'], epoch_df['val_accuracy'], 
                       label=label, alpha=0.8, linewidth=2, color=color,
                       marker=marker, markersize=3, markevery=max(1, len(epoch_df)//20))
        
        # Print summary stats
        print(f"\n{label}:")
        print_summary_stats(df, label)
    
    # Top Left: Training Loss vs Wall Time
    axes[0, 0].set_xlabel('Wall Time (minutes)')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss vs Wall Time (Per Epoch)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top Right: Training Loss vs Epochs
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Training Loss')
    axes[0, 1].set_title('Training Loss vs Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom Left: Validation Loss vs Wall Time
    axes[1, 0].set_xlabel('Wall Time (minutes)')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Validation Loss vs Wall Time (Per Epoch)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom Right: Validation Accuracy vs Wall Time
    axes[1, 1].set_xlabel('Wall Time (minutes)')
    axes[1, 1].set_ylabel('Validation Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy vs Wall Time (Per Epoch)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        output_path = Path(output_dir) / "experiment_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument('log_files', nargs='+', help="Path to log files (CSV or JSON)")
    parser.add_argument('--compare', action='store_true', 
                       help="Compare multiple experiments")
    parser.add_argument('--output-dir', type=str, 
                       help="Directory to save plots")
    parser.add_argument('--labels', nargs='+', 
                       help="Labels for experiments (when comparing)")
    parser.add_argument('--max-epochs', type=int,
                       help="Maximum number of epochs to plot (truncates data)")
    
    args = parser.parse_args()
    
    if args.compare and len(args.log_files) > 1:
        compare_experiments(args.log_files, args.labels, args.output_dir, args.max_epochs)
    else:
        # Analyze single experiment
        for log_file in args.log_files:
            log_path = Path(log_file)
            experiment_name = log_path.stem.replace('_metrics', '').replace('_log', '')
            
            print(f"Analyzing: {log_file}")
            df = load_log_data(log_file)
            
            print_summary_stats(df, experiment_name)
            plot_training_curves(df, args.output_dir, experiment_name, args.max_epochs)


if __name__ == "__main__":
    main()