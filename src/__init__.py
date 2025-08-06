"""
NoProp: Training Neural Networks Without Back-propagation or Forward-propagation

This package implements the NoProp method from the paper for training neural networks
using denoising score matching instead of traditional backpropagation.
"""

from .models import NoPropNetwork, DenoisingModule
from .trainer import NoPropTrainer
from .config import NoPropConfig, MNISTConfig, CIFAR10Config, CIFAR100Config, get_config
from .dataloaders import load_dataset, get_dataset_info, DATASET_LOADERS
from .utils import set_seed, get_device, print_device_info, Timer, AverageMeter

__version__ = "1.0.0"
__author__ = "NoProp Implementation"

__all__ = [
    # Core classes
    'NoPropNetwork',
    'DenoisingModule', 
    'NoPropTrainer',
    
    # Configuration
    'NoPropConfig',
    'MNISTConfig',
    'CIFAR10Config', 
    'CIFAR100Config',
    'get_config',
    
    # Data loading
    'load_dataset',
    'get_dataset_info',
    'DATASET_LOADERS',
    
    # Utilities
    'set_seed',
    'get_device',
    'print_device_info',
    'Timer',
    'AverageMeter',
]