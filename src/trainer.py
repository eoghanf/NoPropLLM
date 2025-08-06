"""
Core NoProp training logic.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import time

from .models import NoPropNetwork
from .config import NoPropConfig
from .dataloaders import get_dataset_info
from .utils import Timer, AverageMeter, save_checkpoint, print_model_info


class NoPropTrainer:
    """
    NoProp trainer that handles the complete training pipeline.
    """
    
    def __init__(self, config: NoPropConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get dataset information
        self.dataset_info = get_dataset_info(config.dataset)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizers
        self.optimizers = self._create_optimizers()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # Timers
        self.epoch_timer = Timer()
        
    def _create_model(self) -> NoPropNetwork:
        """Create and initialize the NoProp model."""
        model = NoPropNetwork(
            num_layers=self.config.num_layers,
            image_channels=self.dataset_info['image_channels'],
            image_size=self.dataset_info['image_size'],
            label_dim=self.dataset_info['num_classes'],
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        return model
    
    def _create_optimizers(self) -> Dict:
        """Create optimizers for each component."""
        optimizers = {}
        
        # Create separate optimizers for each denoising module
        for layer_idx in range(self.model.num_layers):
            optimizer = optim.AdamW(
                self.model.denoising_modules[layer_idx].parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            optimizers[f'layer_{layer_idx}'] = optimizer
        
        # Optimizer for final classifier and embedding matrix
        final_optimizer = optim.AdamW([
            {'params': self.model.classifier.parameters()},
            {'params': self.model.embed_matrix, 'lr': self.config.learning_rate * self.config.embed_lr_multiplier}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        optimizers['final'] = final_optimizer
        
        return optimizers
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        loss_meter = AverageMeter()
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
            
            # Train final classifier every batch (as per Equation 8)
            self.optimizers['final'].zero_grad()
            
            # Compute classifier loss E[−log p̂_θout(y|z_T)]
            classifier_loss = self.model.compute_classifier_loss(data, target)
            
            if not torch.isnan(classifier_loss):
                classifier_loss.backward()
                
                # Gradient clipping for classifier
                torch.nn.utils.clip_grad_norm_(
                    self.model.classifier.parameters(), 
                    max_norm=self.config.grad_clip_max_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    [self.model.embed_matrix], 
                    max_norm=self.config.grad_clip_max_norm
                )
                
                self.optimizers['final'].step()
                batch_loss += classifier_loss.item()
            
            loss_meter.update(batch_loss, data.size(0))
            
            # Log training progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {batch_loss:.6f}')
        
        return loss_meter.avg
    
    def validate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model and return loss and accuracy.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (test_loss, accuracy)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
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
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
        """
        print("Starting training...")
        print("=" * 80)
        
        # Initial validation
        test_loss, accuracy = self.validate(test_loader)
        print(f"Before training, model test loss is {test_loss} and accuracy is {accuracy:.2f}%")
        
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            self.epoch_timer.start()
            
            # Train for one epoch
            avg_train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            test_loss, accuracy = self.validate(test_loader)
            
            # Track best accuracy and save model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                if self.config.save_best:
                    self.save_checkpoint(self.config.best_model_path, is_best=True)
            
            epoch_time = self.epoch_timer.stop()
            
            # Log results
            print(f'Epoch {epoch:3d}/{self.config.epochs} | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Test Loss: {test_loss:.4f} | '
                  f'Accuracy: {accuracy:.2f}% | '
                  f'Time: {epoch_time:.1f}s')
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'accuracy': accuracy,
                'time': epoch_time
            })
            
            # Early stopping if accuracy is very high
            if self.config.early_stopping and accuracy >= self.config.early_stopping_accuracy:
                print(f"Early stopping at epoch {epoch} with accuracy {accuracy:.2f}%")
                break
        
        print("=" * 80)
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        # Save final model
        if self.config.save_final:
            self.save_checkpoint(self.config.final_model_path, is_best=False)
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint."""
        # Convert layer optimizers dict for saving
        layer_optimizers = {}
        for key, optimizer in self.optimizers.items():
            if key.startswith('layer_'):
                layer_idx = int(key.split('_')[1])
                layer_optimizers[layer_idx] = optimizer.state_dict()
            else:
                layer_optimizers[key] = optimizer.state_dict()
        
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizers': layer_optimizers,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'config': self.config,
            'dataset_info': self.dataset_info
        }
        
        save_checkpoint(
            self.model, 
            {}, # optimizers handled above
            self.current_epoch,
            self.best_accuracy,
            filepath,
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
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded from '{filepath}' (epoch {self.current_epoch}, accuracy {self.best_accuracy:.2f}%)")
        
        return checkpoint
    
    def print_info(self):
        """Print trainer information."""
        print(f"Dataset: {self.config.dataset.upper()}")
        print(f"Model: NoProp with {self.config.num_layers} layers")
        print_model_info(self.model)
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print()