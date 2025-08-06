import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from typing import Dict, Tuple
import time

from src.models import NoPropNetwork


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset with standard normalization."""
    # MNIST normalization (mean=0.1307, std=0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders - optimized for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    
    return train_loader, test_loader


def train_epoch(model: NoPropNetwork, train_loader: DataLoader, 
                optimizers: Dict[int, optim.Optimizer], final_optimizer: optim.Optimizer,
                device: torch.device, epoch: int, log_interval: int = 100) -> float:
    """
    Train the NoProp model for one epoch.
    Each denoising module is trained independently.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        batch_size = data.shape[0]
        
        # Train each denoising module independently
        epoch_loss = 0.0
        for layer_idx in range(model.num_layers):
            optimizers[layer_idx].zero_grad()
            
            # Compute loss for this layer (clean image + layer-specific noisy labels)
            loss_layer = model.compute_loss(data, target, layer_idx)
            
            # Skip if loss is NaN 
            if torch.isnan(loss_layer):
                continue
                
            loss_layer.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.denoising_modules[layer_idx].parameters(), max_norm=1.0)
            
            optimizers[layer_idx].step()
            epoch_loss += loss_layer.item()
        
        # Train final classifier every batch (as per Equation 8)
        final_optimizer.zero_grad()
        
        # Compute classifier loss E[−log p̂_θout(y|z_T)]
        classifier_loss = model.compute_classifier_loss(data, target)
        
        if not torch.isnan(classifier_loss):
            classifier_loss.backward()
            
            # Gradient clipping for classifier
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([model.embed_matrix], max_norm=1.0)
            
            final_optimizer.step()
            epoch_loss += classifier_loss.item()
        
        total_loss += epoch_loss
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {epoch_loss:.6f}')
    
    return total_loss / num_batches


def validate_model(model: NoPropNetwork, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Validate the NoProp model and return cross-entropy loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Forward pass through inference pipeline
            log_probs = model(data, mode='inference')  # Returns log probabilities
            
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


def main():
    # Set hyperparameters from Table 3 (MNIST row)
    batch_size = 128
    epochs = 100
    learning_rate = 0.001
    weight_decay = 0.001
    timesteps = 10  # T = 10
    eta = 0.1  # η hyperparameter
    
    print("=== NoProp MNIST Training ===")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Timesteps: {timesteps}")
    print(f"Eta (η): {eta}")
    print()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Initialize model
    print("Initializing NoProp network...")
    model = NoPropNetwork(
        num_layers=timesteps,
        image_channels=1,  # MNIST is grayscale
        image_size=28,     # MNIST is 28x28
        label_dim=10,      # 10 classes
        hidden_dim=256
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create separate optimizers for each denoising module (as mentioned in paper)
    optimizers = {}
    for layer_idx in range(model.num_layers):
        # Each module gets its own AdamW optimizer
        optimizer = optim.AdamW(
            model.denoising_modules[layer_idx].parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        optimizers[layer_idx] = optimizer
    
    # Also optimizer for final classifier and embedding matrix
    final_optimizer = optim.AdamW([
        {'params': model.classifier.parameters()},
        {'params': model.embed_matrix, 'lr': learning_rate * 0.1}  # Lower LR for embeddings
    ], lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    print("Starting training...")
    print("=" * 80)
    
    best_accuracy = 0.0
    training_history = []

    test_loss, accuracy = validate_model(model, test_loader, device)
    print(f"Before training, model test loss is {test_loss} and accuracy is {accuracy}")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train for one epoch
        avg_train_loss = train_epoch(model, train_loader, optimizers, final_optimizer, device, epoch)
        
        # Final classifier is now trained every batch in train_epoch()
        
        # Validate
        test_loss, accuracy = validate_model(model, test_loader, device)
        
        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizers': {layer_idx: opt.state_dict() for layer_idx, opt in optimizers.items()},
                'best_accuracy': best_accuracy,
            }, 'best_mnist_noprop_model.pt')
        
        epoch_time = time.time() - start_time
        
        # Log results
        print(f'Epoch {epoch:3d}/{epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Test Loss: {test_loss:.4f} | '
              f'Accuracy: {accuracy:.2f}% | '
              f'Time: {epoch_time:.1f}s')
        
        # Store history
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy
        })
        
        # Early stopping if accuracy is very high
        if accuracy >= 99.5:
            print(f"Early stopping at epoch {epoch} with accuracy {accuracy:.2f}%")
            break
    
    print("=" * 80)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    # Save final model and training history
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'hyperparameters': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'timesteps': timesteps,
            'eta': eta
        }
    }, 'final_mnist_noprop_model.pt')
    
    print("Model saved to 'final_mnist_noprop_model.pt'")
    print("Best model saved to 'best_mnist_noprop_model.pt'")


if __name__ == "__main__":
    main()