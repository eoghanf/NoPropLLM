# Log Analysis Tools

This directory contains tools for analyzing and visualizing training logs from experiments.

## compare_logs.py

The main analysis script that provides visualization and comparison of training experiments.

### Features

- **Single Experiment Analysis**: Plot training curves for individual experiments
- **Multi-Experiment Comparison**: Compare multiple experiments side-by-side
- **2x2 Plot Layout**: Structured visualization with consistent axes
- **Summary Statistics**: Print detailed experiment statistics
- **Export Options**: Save plots to files for reports/papers

### Plot Layout

The script generates a 2x2 plot layout with the following structure:

```
┌─────────────────────┬─────────────────────┐
│  Training Loss      │  Training Loss      │
│  vs Wall Time       │  vs Epochs          │
│  (minutes)          │                     │
├─────────────────────┼─────────────────────┤
│  Validation Loss    │  Validation Accuracy│
│  vs Wall Time       │  vs Wall Time       │
│  (minutes)          │  (minutes)          │
└─────────────────────┴─────────────────────┘
```

- **Left Column**: Wall time on X-axis (useful for comparing training efficiency)
- **Top Row**: Training loss (blue lines)
- **Bottom Left**: Validation loss (red lines) vs wall time
- **Bottom Right**: Validation accuracy (green lines) vs wall time

### Usage

#### Single Experiment Analysis
```bash
python log_analysis/compare_logs.py training_logs/mnist_diffusion_*_metrics.csv
```

This will:
1. Load the specified log file
2. Display summary statistics
3. Show the 2x2 training curve plot

#### Multi-Experiment Comparison
```bash
python log_analysis/compare_logs.py \
  training_logs/mnist_diffusion_*_metrics.csv \
  training_logs/mnist_backpropagation_*_metrics.csv \
  --compare \
  --labels "NoProp Diffusion" "Standard Backprop"
```

This will:
1. Load multiple log files
2. Display summary statistics for each experiment
3. Show comparative plots with different colored lines for each experiment

#### Save Plots to File
```bash
python log_analysis/compare_logs.py \
  training_logs/mnist_*_metrics.csv \
  --compare \
  --output-dir plots/
```

### Command Line Options

- `log_files`: One or more paths to CSV or JSON log files
- `--compare`: Enable comparison mode for multiple experiments
- `--labels`: Custom labels for experiments (when comparing)
- `--output-dir`: Directory to save plot files
- `--max-epochs`: Maximum number of epochs to plot (truncates data at specified epoch)

#### Max Epochs Feature

The `--max-epochs` argument allows you to focus on early training behavior by truncating all plots at a specified epoch number. This affects both time-based and epoch-based plots:

- **Epoch-based plots** (right column): X-axis limited to max_epochs
- **Time-based plots** (left column): Only shows wall-clock time up to the specified epoch
- **Data filtering**: All metrics (loss, accuracy, etc.) are filtered to the epoch range
- **Multiple experiments**: When comparing, all experiments are truncated at the same epoch

This is particularly useful for:
- Analyzing early convergence behavior
- Comparing training methods in the initial epochs
- Focusing on rapid training phases
- Creating consistent comparisons across experiments with different total epochs

### Supported File Formats

- **CSV files** (`.csv`): Primary format, fastest loading
- **JSON files** (`.json`): Complete experiment logs with metadata

### Examples

#### Compare Different Datasets
```bash
python log_analysis/compare_logs.py \
  training_logs/mnist_diffusion_*_metrics.csv \
  training_logs/cifar10_diffusion_*_metrics.csv \
  training_logs/cifar100_diffusion_*_metrics.csv \
  --compare \
  --labels "MNIST" "CIFAR-10" "CIFAR-100"
```

#### Compare Training Methods
```bash
python log_analysis/compare_logs.py \
  training_logs/cifar10_diffusion_*_metrics.csv \
  training_logs/cifar10_backpropagation_*_metrics.csv \
  --compare \
  --labels "Diffusion Training" "Backpropagation Training" \
  --output-dir comparison_plots/
```

#### Focus on Early Training (Max Epochs)
```bash
# Compare only the first 10 epochs (useful for analyzing early convergence)
python log_analysis/compare_logs.py \
  training_logs/mnist_diffusion_*_metrics.csv \
  training_logs/mnist_backpropagation_*_metrics.csv \
  --compare \
  --labels "NoProp Diffusion" "Standard Backprop" \
  --max-epochs 10
```

#### Single Experiment with Epoch Limit
```bash
# Plot only first 5 epochs of training
python log_analysis/compare_logs.py \
  training_logs/cifar10_diffusion_*_metrics.csv \
  --max-epochs 5
```

### Output Files

When using `--output-dir`:
- **Single experiment**: `{experiment_name}_training_curves.png`
- **Multiple experiments**: `experiment_comparison.png`

### Dependencies

- `pandas`: Data loading and manipulation
- `matplotlib`: Plotting and visualization
- `numpy`: Numerical operations
- `pathlib`: File path handling

Install with:
```bash
pip install pandas matplotlib numpy
```