#!/usr/bin/env python3
"""
Clean MNIST training script using YAML configuration.

Usage:
    python train_mnist.py                          # Use default experiment_configs/mnist.yaml
    python train_mnist.py --config my_config.yaml # Use custom config file
    python train_mnist.py --epochs 50             # Override specific parameters
"""

import argparse
from pathlib import Path

from src import (
    NoPropTrainer,
    load_config,
    load_dataset,
    set_seed,
    print_device_info,
    print_training_header,
    list_available_configs
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NoProp model on MNIST dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help="Path to YAML configuration file (default: experiment_configs/mnist.yaml)"
    )
    
    # Override parameters
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, help="Weight decay")
    parser.add_argument('--num_layers', type=int, help="Number of denoising layers")
    parser.add_argument('--hidden_dim', type=int, help="Hidden dimension size")
    parser.add_argument('--seed', type=int, help="Random seed")
    
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
            print(f"\nUse with: python train_mnist.py --config experiment_configs/<name>.yaml")
        else:
            print("  No configuration files found in experiment_configs/")
        return
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(config_path=args.config)
    else:
        print("Using default MNIST configuration")
        config = load_config(dataset='mnist')
    
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
        augment=getattr(config, 'augment', False)
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


if __name__ == "__main__":
    main()