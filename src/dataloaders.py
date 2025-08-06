"""
Data loading utilities for NoProp training.
Supports MNIST, CIFAR-10, CIFAR-100, and extensible to custom datasets.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


def load_mnist_data(batch_size: int = 128, data_path: str = './data', 
                   num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset with standard normalization.
    
    Args:
        batch_size: Batch size for data loaders
        data_path: Path to store/load data
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # MNIST normalization (mean=0.1307, std=0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_path, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders - optimized for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


def load_cifar10_data(batch_size: int = 128, data_path: str = './data',
                     num_workers: int = 4, augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        data_path: Path to store/load data
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation to training set
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # CIFAR-10 normalization
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # Training transform (with optional augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


def load_cifar100_data(batch_size: int = 128, data_path: str = './data',
                      num_workers: int = 4, augment: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-100 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        data_path: Path to store/load data
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation to training set
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # CIFAR-100 uses same normalization as CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # Training transform (with optional augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


# Registry for easy access
DATASET_LOADERS = {
    'mnist': load_mnist_data,
    'cifar10': load_cifar10_data,
    'cifar100': load_cifar100_data,
}


def get_dataset_info(dataset_name: str) -> dict:
    """Get dataset-specific information."""
    info = {
        'mnist': {
            'image_channels': 1,
            'image_size': 28,
            'num_classes': 10,
            'input_shape': (1, 28, 28)
        },
        'cifar10': {
            'image_channels': 3,
            'image_size': 32,
            'num_classes': 10,
            'input_shape': (3, 32, 32)
        },
        'cifar100': {
            'image_channels': 3,
            'image_size': 32,
            'num_classes': 100,
            'input_shape': (3, 32, 32)
        }
    }
    
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(info.keys())}")
    
    return info[dataset_name]


def load_dataset(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load dataset by name with unified interface.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'cifar100')
        **kwargs: Additional arguments passed to dataset loader
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_LOADERS.keys())}")
    
    loader_fn = DATASET_LOADERS[dataset_name]
    return loader_fn(**kwargs)