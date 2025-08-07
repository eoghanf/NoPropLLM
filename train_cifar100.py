#!/usr/bin/env python3
"""
Clean CIFAR-100 training script using YAML configuration.

Usage:
    python train_cifar100.py                          # Use default experiment_configs/cifar100.yaml
    python train_cifar100.py --config my_config.yaml # Use custom config file
    python train_cifar100.py --epochs 250            # Override specific parameters
"""

import argparse
from pathlib import Path

from src import (
    NoPropTrainer,
    load_dataset,
    set_seed,
    print_device_info,
)
from src.config import load_config, list_available_configs
from src.utils import print_training_header


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NoProp model on CIFAR-100 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help="Path to YAML configuration file (default: experiment_configs/cifar100.yaml)"
    )
    
    # Override parameters
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, help="Weight decay")
    parser.add_argument('--num_layers', type=int, help="Number of denoising layers")
    parser.add_argument('--hidden_dim', type=int, help="Hidden dimension size")
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--augment', action='store_true', help="Enable data augmentation")
    parser.add_argument('--log_interval', type=int, help="Log interval for training progress")
    parser.add_argument('--timesteps', type=int, help="Number of timesteps for NoProp")
    parser.add_argument('--eta', type=float, help="Eta parameter for NoProp")
    parser.add_argument('--grad_clip_max_norm', type=float, help="Gradient clipping max norm")
    
    # Utility options
    parser.add_argument(
        '--list-configs', 
        action='store_true',
        help="List all available configuration files"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # List available configs if requested
    if args.list_configs:
        print("Available configuration files:")
        configs = list_available_configs()
        if configs:
            for config in configs:
                print(f"  - {config}")
            print(f"\nUse with: python train_cifar100.py --config experiment_configs/<name>.yaml")
        else:
            print("  No configuration files found in experiment_configs/")
        return
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(config_path=args.config)
    else:
        print("Using default CIFAR-100 configuration")
        config = load_config(dataset='cifar100')
    
    # Override configuration parameters from command line
    overrides = {}
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'list_configs']:
            overrides[key] = value
    
    if overrides:
        print(f"Overriding config parameters: {overrides}")
        config = config.update(**overrides)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Print configuration and device info
    print_training_header(config)
    print_device_info()
    
    # Load dataset
    print(f"Loading {config.dataset.upper()} dataset...")
    train_loader, test_loader = load_dataset(
        config.dataset,
        batch_size=config.batch_size,
        data_path=config.data_path,
        num_workers=config.num_workers,
        augment=config.augment
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Create trainer
    trainer = NoPropTrainer(config)
    trainer.print_info()
    
    # Train the model
    trainer.train(train_loader, test_loader)
    
    print(f"Training completed!")
    print(f"Best model saved to: {config.best_model_path}")
    print(f"Final model saved to: {config.final_model_path}")
    
    # Print final results summary
    if trainer.training_history:
        best_accuracy = max(entry['accuracy'] for entry in trainer.training_history)
        print(f"Best accuracy achieved: {best_accuracy:.2f}%")
        
        # Print paper comparison (typical CIFAR-100 results)
        print("\n=== Results Summary ===")
        print(f"CIFAR-100 Best Accuracy: {best_accuracy:.2f}%")
        print("Note: CIFAR-100 is a challenging dataset with 100 classes")
        print("Good results for NoProp on CIFAR-100 are typically 60-70%")


if __name__ == "__main__":
    main()