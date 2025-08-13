#!/usr/bin/env python3
"""
Quick test of the logging system.
"""

import torch
from pathlib import Path
from src.config import load_config
from src.dataloaders import load_dataset
from src.trainer import NoPropTrainer
from src.utils import set_seed

def test_logging():
    """Test the logging system with a short run."""
    print("Testing logging system...")
    
    # Load MNIST config and override for quick test
    config_path = Path("experiment_configs/mnistdiffusion.yaml")
    config = load_config(config_path=config_path)
    
    # Override for quick test
    config = config.update(
        epochs=1,  # Just 1 epoch
        batch_size=64,  # Small batch size
        log_interval=1,  # Log every batch
        early_stopping=False,
        validation_batches_per_log=2  # Use fewer validation batches
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_dataset(
        config.dataset,
        batch_size=config.batch_size,
        data_path=config.data_path,
        num_workers=2  # Fewer workers for test
    )
    
    # Create trainer
    trainer = NoPropTrainer(config)
    trainer.print_info()
    
    print("\nStarting test training (just a few batches)...")
    
    # Train for just a few batches by limiting the data loader
    limited_train_data = []
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Only process 5 batches
            break
        limited_train_data.append(batch)
    
    # Manual training loop for test
    trainer.model.train()
    epoch = 1
    
    for batch_idx, (data, target) in enumerate(limited_train_data):
        trainer.batch_timer.start()
        
        data = data.to(trainer.device)
        target = target.to(trainer.device)
        
        if trainer.training_mode == 'diffusion':
            # Simplified diffusion training for test
            batch_loss = 0.0
            for layer_idx in range(min(3, trainer.model.num_layers)):  # Only first 3 layers
                loss_layer = trainer.model.compute_loss(data, target, layer_idx)
                if not torch.isnan(loss_layer):
                    batch_loss += loss_layer.item()
        else:
            # Simplified backprop training
            output = trainer.model(data, mode='inference')
            loss = torch.nn.functional.nll_loss(output, target)
            batch_loss = loss.item()
        
        # Compute validation metrics
        batch_time = trainer.batch_timer.stop()
        val_loss, val_accuracy = trainer._validate_batch_subset(test_loader, max_batches=2)
        
        # Log metrics
        trainer.logger.log_batch_metrics(
            epoch=epoch,
            batch_idx=batch_idx,
            train_loss=batch_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=trainer._get_current_lr(),
            batch_size=data.size(0),
            batch_time=batch_time
        )
        
        print(f"Batch {batch_idx}: Train Loss: {batch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
              f"Time: {batch_time:.3f}s")
    
    # Finalize logging
    trainer.logger.log_epoch_end(epoch)
    trainer.logger.finalize()
    
    # Print summary
    summary = trainer.logger.get_experiment_summary()
    print("\n=== Test Summary ===")
    for key, value in summary.items():
        if key != 'log_files':
            print(f"{key}: {value}")
    
    print(f"\nLog files created:")
    for file_type, path in summary['log_files'].items():
        print(f"  {file_type}: {path}")
    
    print("\nLogging test completed successfully!")


if __name__ == "__main__":
    test_logging()