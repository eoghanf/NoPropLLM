"""
Core NoProp training logic.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional

from .models import NoPropNetwork
from .config import NoPropConfig
from .dataloaders import get_dataset_info
from .utils import Timer, AverageMeter, save_checkpoint as save_checkpoint_util, print_model_info
from .logger import TrainingLogger, BatchTimer


class NoPropTrainer:
    """
    Unified trainer that handles both diffusion (NoProp) and backpropagation training.
    """
    
    def __init__(self, config: NoPropConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_mode = getattr(config, 'training', 'diffusion')  # Default to diffusion for backward compatibility
        
        # Get dataset information
        self.dataset_info = get_dataset_info(config.dataset)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizers based on training mode
        if self.training_mode == 'diffusion':
            self.optimizers = self._create_diffusion_optimizers()
        else:
            self.optimizer = self._create_backprop_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # Loss collection for statistics
        self.epoch_losses = []
        
        # Timers
        self.epoch_timer = Timer()
        self.batch_timer = BatchTimer()
        
        # Initialize logger
        experiment_name = f"{config.dataset}_{self.training_mode}"
        self.logger = TrainingLogger(
            experiment_name=experiment_name,
            log_dir="training_logs",
            config=vars(config) if hasattr(config, '__dict__') else config._asdict() if hasattr(config, '_asdict') else {}
        )
        
    def _create_model(self) -> NoPropNetwork:
        """Create and initialize the NoProp model."""
        model = NoPropNetwork(
            num_layers=self.config.num_layers,
            image_channels=self.dataset_info['image_channels'],
            image_size=self.dataset_info['image_size'],
            label_dim=self.dataset_info['num_classes'],
            noise_schedule_type=self.config.noise_schedule_type,
            noise_schedule_min=self.config.noise_schedule_min,
            noise_schedule_max=self.config.noise_schedule_max
        ).to(self.device)
        
        return model
    
    def _create_diffusion_optimizers(self) -> Dict:
        """Create optimizers for each denoising layer (classifier is identity - no parameters)."""
        optimizers = {}
        
        # Create separate optimizers for each denoising module
        for layer_idx in range(self.model.num_layers):
            optimizer = optim.AdamW(
                self.model.denoising_modules[layer_idx].parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            optimizers[f'layer_{layer_idx}'] = optimizer
        
        # No optimizer needed for final classifier (identity function has no parameters)
        
        return optimizers
    
    def _create_backprop_optimizer(self):
        """Create standard backpropagation optimizer."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer
    
    def train_epoch(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """
        Train the model for one epoch using either diffusion or backpropagation.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader (for batch-level validation)
            epoch: Current epoch number
            
        Returns:
            Tuple of (average training loss, epoch statistics dict)
        """
        if self.training_mode == 'diffusion':
            return self._train_epoch_diffusion(train_loader, test_loader, epoch)
        else:
            return self._train_epoch_backprop(train_loader, test_loader, epoch)
    
    def _train_epoch_diffusion(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int) -> float:
        """Train the model for one epoch using diffusion (NoProp) training."""
        self.model.train()
        loss_meter = AverageMeter()
        self.epoch_losses = []  # Reset for new epoch
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Start batch timer
            self.batch_timer.start()
            
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            batch_loss = 0.0
            
            # Train each denoising module independently
            for layer_idx in range(self.model.num_layers):
                self.optimizers[f'layer_{layer_idx}'].zero_grad()
                
                # Compute loss for this layer (clean image + layer-specific noisy labels)
                loss_layer = self.model.compute_loss(data, target, layer_idx)
                
                # Skip if loss is NaN
                if torch.isnan(loss_layer):
                    continue
                
                loss_layer.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.denoising_modules[layer_idx].parameters(), 
                    max_norm=self.config.grad_clip_max_norm
                )
                
                self.optimizers[f'layer_{layer_idx}'].step()
                batch_loss += loss_layer.item()
            
            # No separate classifier training needed (identity function has no parameters)
            # The final denoising layer serves as the classifier
            
            # Collect loss for epoch statistics
            self.epoch_losses.append(batch_loss)
            loss_meter.update(batch_loss, data.size(0))
            
            # Stop batch timer 
            batch_time = self.batch_timer.stop()
            
            # For diffusion training, skip batch-level validation as it's unreliable
            # (model is in inconsistent state during layer-by-layer training)
            if self.training_mode == 'diffusion':
                # Use dummy values for batch-level logging during diffusion
                val_loss, val_accuracy = 0.0, 0.0
            else:
                # For backprop, batch-level validation is fine since whole model is trained
                val_batches = getattr(self.config, 'validation_batches_per_log', 5)
                val_loss, val_accuracy = self._validate_batch_subset(test_loader, max_batches=val_batches)
            
            # Log metrics after every batch
            self.logger.log_batch_metrics(
                epoch=epoch,
                batch_idx=batch_idx,
                train_loss=batch_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=self._get_current_lr(),
                batch_size=data.size(0),
                batch_time=batch_time
            )
            
            # Log training progress to console
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {batch_loss:.6f}\tVal Acc: {val_accuracy:.2f}%')
        
        # At epoch end, compute and log statistics
        epoch_stats = self._compute_epoch_loss_stats()
        if epoch_stats:
            print(f'Epoch {epoch} Loss Stats - Mean: {epoch_stats["train_loss_mean"]:.6f}, '
                  f'Std: {epoch_stats["train_loss_std"]:.6f}, '
                  f'Min: {epoch_stats["train_loss_min"]:.6f}, '
                  f'Max: {epoch_stats["train_loss_max"]:.6f}')
        
        # Return epoch stats so main training loop can log with validation metrics
        return loss_meter.avg, epoch_stats
    
    def _train_epoch_backprop(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int) -> float:
        """Train the model for one epoch using standard backpropagation."""
        self.model.train()
        loss_meter = AverageMeter()
        self.epoch_losses = []  # Reset for new epoch
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Start batch timer
            self.batch_timer.start()
            
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass through the model for inference
            output = self.model(data, mode='inference')
            
            # Compute cross-entropy loss
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_max_norm
            )
            
            self.optimizer.step()
            
            # Collect loss for epoch statistics
            self.epoch_losses.append(loss.item())
            loss_meter.update(loss.item(), data.size(0))
            
            # Stop batch timer and compute validation metrics
            batch_time = self.batch_timer.stop()
            val_batches = getattr(self.config, 'validation_batches_per_log', 5)
            val_loss, val_accuracy = self._validate_batch_subset(test_loader, max_batches=val_batches)
            
            # Log metrics after every batch
            self.logger.log_batch_metrics(
                epoch=epoch,
                batch_idx=batch_idx,
                train_loss=loss.item(),
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=self._get_current_lr(),
                batch_size=data.size(0),
                batch_time=batch_time
            )
            
            # Log training progress to console
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\tVal Acc: {val_accuracy:.2f}%')
        
        # At epoch end, compute and log statistics
        epoch_stats = self._compute_epoch_loss_stats()
        if epoch_stats:
            print(f'Epoch {epoch} Loss Stats - Mean: {epoch_stats["train_loss_mean"]:.6f}, '
                  f'Std: {epoch_stats["train_loss_std"]:.6f}, '
                  f'Min: {epoch_stats["train_loss_min"]:.6f}, '
                  f'Max: {epoch_stats["train_loss_max"]:.6f}')
        
        # Return epoch stats so main training loop can log with validation metrics
        return loss_meter.avg, epoch_stats
    
    def _validate_batch_subset(self, test_loader: DataLoader, max_batches: int = 5) -> Tuple[float, float]:
        """
        Quick validation on a subset of validation data for frequent logging.
        
        Args:
            test_loader: Test data loader
            max_batches: Maximum number of batches to evaluate (for speed)
            
        Returns:
            Tuple of (test_loss, accuracy)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx >= max_batches:
                    break
                    
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Forward pass through inference pipeline
                log_probs = self.model(data, mode='inference')
                
                # Compute cross-entropy loss
                loss = F.nll_loss(log_probs, target, reduction='sum')
                test_loss += loss.item()
                
                # Get predictions
                pred = log_probs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= total
        accuracy = 100. * correct / total
        
        # Return to training mode
        self.model.train()
        
        return test_loss, accuracy
    
    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer(s)."""
        if self.training_mode == 'diffusion':
            # Get LR from the first layer optimizer (they should all be the same)
            return self.optimizers['layer_0'].param_groups[0]['lr']
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def _compute_epoch_loss_stats(self) -> Dict:
        """Compute epoch-level loss statistics from collected batch losses."""
        if not self.epoch_losses:
            return {}
        
        losses = torch.tensor(self.epoch_losses)
        return {
            'train_loss_mean': losses.mean().item(),
            'train_loss_std': losses.std().item(),
            'train_loss_min': losses.min().item(), 
            'train_loss_max': losses.max().item(),
            'train_loss_median': losses.median().item(),
            'train_loss_count': len(self.epoch_losses)
        }
    
    def _evaluate_epoch_0(self, train_loader: DataLoader, test_loader: DataLoader):
        """Evaluate untrained model performance and log as Epoch 0."""
        print("\nEvaluating untrained model (Epoch 0)...")
        
        # Get validation performance
        test_loss, test_accuracy, train_loss, train_accuracy = self.validate(test_loader, train_loader)
        
        print(f"Epoch 0 (Untrained Model):")
        print(f"  Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.2f}%")
        if train_loss is not None and train_accuracy is not None:
            print(f"  Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.2f}%")
        
        # Create epoch 0 statistics (no batch-level variance for untrained model)
        epoch_0_stats = {
            'train_loss_mean': train_loss if train_loss is not None else 0.0,
            'train_loss_std': 0.0,  # No training occurred, so no variance
            'train_loss_min': train_loss if train_loss is not None else 0.0,
            'train_loss_max': train_loss if train_loss is not None else 0.0,
            'train_loss_median': train_loss if train_loss is not None else 0.0,
            'train_loss_count': 0
        }
        
        print(f"Epoch 0 Loss Stats - Mean: {epoch_0_stats['train_loss_mean']:.6f}, "
              f"Std: {epoch_0_stats['train_loss_std']:.6f} (baseline)")
        
        # Log epoch 0 statistics
        self.logger.log_epoch_end_stats(0, epoch_0_stats, test_loss, test_accuracy)
        
        return test_loss, test_accuracy, train_loss, train_accuracy
    
    def validate(self, test_loader: DataLoader, train_loader: DataLoader = None) -> Tuple[float, float, float, float]:
        """
        Validate the model and return loss and accuracy for both validation and training sets.
        
        Args:
            test_loader: Test/validation data loader
            train_loader: Training data loader (optional)
            
        Returns:
            Tuple of (test_loss, test_accuracy, train_loss, train_accuracy)
            If train_loader is None, train_loss and train_accuracy will be None
        """
        self.model.eval()
        
        # Compute validation metrics
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Forward pass through inference pipeline
                log_probs = self.model(data, mode='inference')  # Returns log probabilities
                
                # Compute cross-entropy loss
                loss = F.nll_loss(log_probs, target, reduction='sum')
                test_loss += loss.item()
                
                # Get predictions
                pred = log_probs.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * test_correct / test_total
        
        # Compute training metrics if train_loader is provided
        train_loss = None
        train_accuracy = None
        
        if train_loader is not None:
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with torch.no_grad():
                for data, target in train_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    
                    # Forward pass through inference pipeline
                    log_probs = self.model(data, mode='inference')  # Returns log probabilities
                    
                    # Compute cross-entropy loss
                    loss = F.nll_loss(log_probs, target, reduction='sum')
                    train_loss += loss.item()
                    
                    # Get predictions
                    pred = log_probs.argmax(dim=1, keepdim=True)
                    train_correct += pred.eq(target.view_as(pred)).sum().item()
                    train_total += target.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_accuracy = 100. * train_correct / train_total
        
        return test_loss, test_accuracy, train_loss, train_accuracy
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
        """
        print("Starting training...")
        print("=" * 80)
        
        # Evaluate and log untrained model performance (Epoch 0)
        test_loss, test_accuracy, train_loss, train_accuracy = self._evaluate_epoch_0(train_loader, test_loader)
        
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            self.epoch_timer.start()
            
            # Train for one epoch
            avg_train_loss, epoch_stats = self.train_epoch(train_loader, test_loader, epoch)
            
            # Validate
            test_loss, test_accuracy, train_loss, train_accuracy = self.validate(test_loader, train_loader)
            
            # Track best accuracy and save model
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                if self.config.save_best:
                    self.save_checkpoint(self.config.best_model_path, is_best=True)
            
            epoch_time = self.epoch_timer.stop()
            
            # Log epoch-end statistics with proper validation metrics
            if epoch_stats:
                self.logger.log_epoch_end_stats(epoch, epoch_stats, test_loss, test_accuracy)
            
            # Log epoch end to the logger
            self.logger.log_epoch_end(epoch)
            
            # Log results
            print(f'Epoch {epoch:3d}/{self.config.epochs} | Time: {epoch_time:.1f}s')
            print(f'  Training Loss: {train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%' if train_loss is not None else f'  Training Loss: {avg_train_loss:.4f}')
            print(f'  Validation Loss: {test_loss:.4f} | Validation Accuracy: {test_accuracy:.2f}%')
            
            # Store history
            history_entry = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'time': epoch_time
            }
            if train_loss is not None and train_accuracy is not None:
                history_entry['train_loss_inference'] = train_loss
                history_entry['train_accuracy'] = train_accuracy
            
            self.training_history.append(history_entry)
            
            # Early stopping if accuracy is very high
            if self.config.early_stopping and test_accuracy >= self.config.early_stopping_accuracy:
                print(f"Early stopping at epoch {epoch} with accuracy {test_accuracy:.2f}%")
                break
        
        print("=" * 80)
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        # Finalize logging
        self.logger.finalize()
        
        # Save final model
        if self.config.save_final:
            self.save_checkpoint(self.config.final_model_path, is_best=False)
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint."""
        # Ensure checkpoint directory exists
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'training_history': self.training_history,
            'config': self.config,
            'dataset_info': self.dataset_info
        }
        
        if self.training_mode == 'diffusion':
            optimizers = self.optimizers
        else:
            optimizers = {'main': self.optimizer}
        
        save_checkpoint_util(
            model=self.model, 
            optimizers=optimizers,
            epoch=self.current_epoch,
            best_accuracy=self.best_accuracy,
            filepath=filepath,
            **checkpoint_data
        )
        
        if is_best:
            print(f"New best model saved to '{filepath}' with accuracy {self.best_accuracy:.2f}%")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer states
        if 'optimizers' in checkpoint:
            if self.training_mode == 'diffusion':
                for key, state in checkpoint['optimizers'].items():
                    if isinstance(key, int):
                        # Old format - layer index
                        optimizer_key = f'layer_{key}'
                        if optimizer_key in self.optimizers:
                            self.optimizers[optimizer_key].load_state_dict(state)
                    else:
                        # New format - string key
                        if key in self.optimizers:
                            self.optimizers[key].load_state_dict(state)
            else:
                # Backpropagation mode
                if 'main' in checkpoint['optimizers']:
                    self.optimizer.load_state_dict(checkpoint['optimizers']['main'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded from '{filepath}' (epoch {self.current_epoch}, accuracy {self.best_accuracy:.2f}%)")
        
        return checkpoint
    
    def print_info(self):
        """Print trainer information."""
        print(f"Dataset: {self.config.dataset.upper()}")
        print(f"Training mode: {self.training_mode.upper()}")
        if self.training_mode == 'diffusion':
            print(f"Model: NoProp with {self.config.num_layers} layers")
        else:
            print(f"Model: Standard neural network with {self.config.num_layers} layers")
        print_model_info(self.model)
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print()