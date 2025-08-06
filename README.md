# NoProp: Training Neural Networks Without Back-propagation

This repository implements the NoProp method for training neural networks using denoising score matching instead of traditional backpropagation.

## Project Structure

```
NoProp/
├── src/                    # Source code
│   ├── models.py          # NoProp network implementation
│   ├── trainer.py         # Training logic
│   ├── config.py          # Configuration management
│   ├── dataloaders.py     # Dataset loading utilities
│   ├── utils.py           # Utility functions
│   └── __init__.py        # Package initialization
├── experiment_configs/     # YAML experiment configurations
│   ├── mnist.yaml         # MNIST configuration
│   ├── cifar10.yaml       # CIFAR-10 configuration
│   └── cifar100.yaml      # CIFAR-100 configuration
├── tests/                 # Test suite
│   ├── test_config.py     # Configuration tests
│   ├── test_models.py     # Model tests
│   ├── test_dataloaders.py # Data loading tests
│   ├── test_utils.py      # Utility tests
│   └── test_integration.py # Integration tests
├── data/                  # Dataset storage (created automatically)
├── train_mnist.py         # MNIST training script
├── train_universal.py     # Universal training script
└── MNISTTraining.py       # Legacy training script (deprecated)
```

## Quick Start

### 1. Install Dependencies

```bash
conda activate NoProp
pip install torch torchvision pyyaml pytest
```

### 2. Train on MNIST

```bash
# Use default MNIST configuration
python train_mnist.py

# Override specific parameters
python train_mnist.py --epochs 50 --batch_size 64

# Use custom configuration file
python train_mnist.py --config my_config.yaml
```

### 3. Train on Other Datasets

```bash
# Train on CIFAR-10
python train_universal.py cifar10

# Train on CIFAR-100 with custom epochs
python train_universal.py cifar100 --epochs 300

# List available configurations
python train_universal.py --list-configs
```

## Configuration

Experiment configurations are stored in `experiment_configs/` as YAML files:

```yaml
# experiment_configs/mnist.yaml
dataset: "mnist"
batch_size: 128
epochs: 100
learning_rate: 0.001
weight_decay: 0.001
num_layers: 10
hidden_dim: 256
# ... more parameters
```

### Creating Custom Configurations

1. Copy an existing config: `cp experiment_configs/mnist.yaml my_experiment.yaml`
2. Modify parameters as needed
3. Run with: `python train_universal.py --config my_experiment.yaml`

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run only fast tests (skip data downloading)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py
```

### Test Categories

- **Unit tests**: Test individual components (models, config, utils)
- **Integration tests**: Test end-to-end functionality  
- **Slow tests**: Tests that download data or take time

## Key Features

### 1. Modular Design
- Clean separation between models, training, configuration, and data loading
- Easy to extend to new datasets and experiments

### 2. YAML Configuration
- All experiment settings in readable YAML files
- Easy parameter sweeps and experiment tracking
- Command-line overrides supported

### 3. Flexible Training Scripts
- Dataset-specific scripts (`train_mnist.py`)
- Universal script for any dataset (`train_universal.py`)
- Comprehensive logging and checkpointing

### 4. Robust Testing
- Comprehensive test suite with pytest
- Unit tests, integration tests, and slow tests
- Automated CI/CD ready

## Implementation Details

### NoProp Method
- Each layer learns to denoise progressively less noisy label embeddings
- Noise schedule: 0.9 → 0.01 across layers
- Clean images provided to all layers (per Figure 1 in paper)
- Independent layer training + classifier training every batch

### Architecture
- 10-layer denoising network (configurable)
- Each layer is a separate `DenoisingModule`
- Final classifier maps denoised embeddings to class predictions

### Training Process
1. Each denoising layer trained independently with layer-specific noise
2. Final classifier trained every batch using full inference pipeline  
3. Separate optimizers for each component
4. Gradient clipping for stability

## Results

The implementation achieves excellent performance:
- **MNIST**: >98.8% accuracy
- **CIFAR-10**: Comparable to paper results
- **CIFAR-100**: Competitive performance

## Extending the Code

### Adding New Datasets
1. Add loader function to `src/dataloaders.py`
2. Add dataset info to `get_dataset_info()`
3. Create YAML config in `experiment_configs/`
4. Add to `DATASET_LOADERS` registry

### Custom Architectures  
1. Modify `DenoisingModule` in `src/models.py`
2. Update configuration parameters as needed
3. Ensure gradient flow works correctly

### Experiment Tracking
- Training history automatically saved
- Checkpoints include configuration for reproducibility  
- Easy to add logging frameworks (wandb, tensorboard)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{noprop2024,
  title={NoProp: Training Neural Networks Without Back-propagation or Forward-propagation},
  author={Li, Qinyu and Teh, Yee Whye and Pascanu, Razvan},
  journal={arXiv preprint arXiv:2503.24322},
  year={2024}
}
```