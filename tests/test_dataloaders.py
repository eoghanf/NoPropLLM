"""
Tests for data loading utilities.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataloaders import (
    load_mnist_data, load_cifar10_data, load_cifar100_data,
    get_dataset_info, load_dataset, DATASET_LOADERS
)


class TestDatasetInfo:
    """Test dataset information utilities."""
    
    def test_mnist_info(self):
        """Test MNIST dataset info."""
        info = get_dataset_info('mnist')
        
        assert info['image_channels'] == 1
        assert info['image_size'] == 28
        assert info['num_classes'] == 10
        assert info['input_shape'] == (1, 28, 28)
    
    def test_cifar10_info(self):
        """Test CIFAR-10 dataset info."""
        info = get_dataset_info('cifar10')
        
        assert info['image_channels'] == 3
        assert info['image_size'] == 32
        assert info['num_classes'] == 10
        assert info['input_shape'] == (3, 32, 32)
    
    def test_cifar100_info(self):
        """Test CIFAR-100 dataset info."""
        info = get_dataset_info('cifar100')
        
        assert info['image_channels'] == 3
        assert info['image_size'] == 32
        assert info['num_classes'] == 100
        assert info['input_shape'] == (3, 32, 32)
    
    def test_unknown_dataset(self):
        """Test error handling for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info('unknown_dataset')


class TestDataLoaders:
    """Test data loader functions."""
    
    @pytest.mark.slow
    def test_mnist_loader(self):
        """Test MNIST data loader (downloads data)."""
        train_loader, test_loader = load_mnist_data(batch_size=32, num_workers=0)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert train_loader.batch_size == 32
        assert test_loader.batch_size == 32
        
        # Test a batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        assert images.shape[1:] == (1, 28, 28)  # MNIST shape
        assert labels.dtype == torch.long
        assert labels.max() < 10  # MNIST has 10 classes
        assert labels.min() >= 0
    
    @pytest.mark.slow
    def test_cifar10_loader(self):
        """Test CIFAR-10 data loader."""
        train_loader, test_loader = load_cifar10_data(batch_size=16, num_workers=0)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Test a batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        assert images.shape[1:] == (3, 32, 32)  # CIFAR-10 shape
        assert labels.max() < 10  # CIFAR-10 has 10 classes
        assert labels.min() >= 0
    
    @pytest.mark.slow
    def test_cifar100_loader(self):
        """Test CIFAR-100 data loader."""
        train_loader, test_loader = load_cifar100_data(batch_size=16, num_workers=0)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Test a batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        assert images.shape[1:] == (3, 32, 32)  # CIFAR-100 shape
        assert labels.max() < 100  # CIFAR-100 has 100 classes
        assert labels.min() >= 0
    
    def test_dataset_registry(self):
        """Test dataset loader registry."""
        assert 'mnist' in DATASET_LOADERS
        assert 'cifar10' in DATASET_LOADERS
        assert 'cifar100' in DATASET_LOADERS
        
        # Test that functions are callable
        assert callable(DATASET_LOADERS['mnist'])
        assert callable(DATASET_LOADERS['cifar10'])
        assert callable(DATASET_LOADERS['cifar100'])
    
    @pytest.mark.slow
    def test_load_dataset_unified_interface(self):
        """Test unified dataset loading interface."""
        # Test MNIST through unified interface
        train_loader, test_loader = load_dataset('mnist', batch_size=32, num_workers=0)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert train_loader.batch_size == 32
    
    def test_load_dataset_unknown(self):
        """Test error handling for unknown dataset in unified interface."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset('unknown_dataset')
    
    @pytest.mark.slow  
    def test_augmentation_flag(self):
        """Test data augmentation flag for CIFAR datasets."""
        # Test without augmentation
        train_loader_no_aug, _ = load_cifar10_data(batch_size=8, num_workers=0, augment=False)
        
        # Test with augmentation
        train_loader_aug, _ = load_cifar10_data(batch_size=8, num_workers=0, augment=True)
        
        # Both should work (can't easily test the actual augmentation without checking transforms)
        assert isinstance(train_loader_no_aug, DataLoader)
        assert isinstance(train_loader_aug, DataLoader)
    
    def test_data_loader_parameters(self):
        """Test various data loader parameters."""
        # Test different batch sizes
        train_loader, _ = load_mnist_data(batch_size=64, num_workers=0)
        assert train_loader.batch_size == 64
        
        # Test different number of workers (but use 0 for testing to avoid multiprocessing issues)
        train_loader, _ = load_mnist_data(batch_size=32, num_workers=0)
        assert train_loader.num_workers == 0


@pytest.fixture(autouse=True)
def suppress_download_warnings():
    """Suppress dataset download warnings during tests."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.datasets")