"""
Configuration management for NoProp training experiments.
Loads configurations from YAML files in experiment_configs/
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union


@dataclass
class NoPropConfig:
    """Configuration class for NoProp training."""
    
    # Dataset settings
    dataset: str = "mnist"
    batch_size: int = 128
    data_path: str = "./data"
    num_workers: int = 4
    augment: bool = False
    
    # Model architecture
    num_layers: int = 10
    hidden_dim: int = 256
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    
    # NoProp specific parameters
    timesteps: int = 10
    eta: float = 0.1
    
    # Optimization settings
    grad_clip_max_norm: float = 1.0
    embed_lr_multiplier: float = 0.1
    
    # Logging and saving
    log_interval: int = 100
    save_best: bool = True
    save_final: bool = True
    best_model_path: str = "checkpoints/best_model.pt"
    final_model_path: str = "checkpoints/final_model.pt"
    
    # Reproducibility
    seed: int = 42
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_accuracy: float = 99.5
    
    # Store original config dict
    _config_dict: Dict[str, Any] = field(default_factory=dict, repr=False)
    _config_path: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timesteps != self.num_layers:
            print(f"Warning: timesteps ({self.timesteps}) != num_layers ({self.num_layers})")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'NoPropConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration object loaded from YAML
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        # Create instance with config values
        instance = cls(**config_dict)
        instance._config_dict = config_dict.copy()
        instance._config_path = str(config_path)
        
        return instance
    
    @classmethod
    def from_dataset(cls, dataset: str) -> 'NoPropConfig':
        """
        Load configuration for a specific dataset.
        
        Args:
            dataset: Dataset name (looks for experiment_configs/{dataset}.yaml)
            
        Returns:
            Configuration object loaded from dataset-specific YAML
        """
        # Look for config file in experiment_configs/
        config_path = Path("experiment_configs") / f"{dataset}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"No configuration found for dataset '{dataset}' at {config_path}")
        
        return cls.from_yaml(config_path)
    
    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict, excluding private fields
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        
        with open(output_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def update(self, **kwargs) -> 'NoPropConfig':
        """
        Create a new config with updated values.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            New configuration object with updated values
        """
        # Get current values as dict
        current_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                current_dict[key] = value
        
        # Update with new values
        current_dict.update(kwargs)
        
        # Create new instance
        new_config = self.__class__(**current_dict)
        new_config._config_dict = self._config_dict.copy()
        new_config._config_path = self._config_path
        
        return new_config
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("=== Configuration ===")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                print(f"{key}: {value}")
        print()


def load_config(config_path: Union[str, Path, None] = None, 
                dataset: Optional[str] = None,
                **overrides) -> NoPropConfig:
    """
    Load configuration from various sources.
    
    Args:
        config_path: Path to specific YAML file (takes precedence)
        dataset: Dataset name to load default config for
        **overrides: Override any configuration values
        
    Returns:
        Configuration object
        
    Raises:
        ValueError: If neither config_path nor dataset is provided
    """
    if config_path is not None:
        config = NoPropConfig.from_yaml(config_path)
    elif dataset is not None:
        config = NoPropConfig.from_dataset(dataset)
    else:
        raise ValueError("Either config_path or dataset must be provided")
    
    # Apply any overrides
    if overrides:
        config = config.update(**overrides)
    
    return config


def list_available_configs() -> list:
    """
    List all available configuration files in experiment_configs/
    
    Returns:
        List of available dataset configurations
    """
    configs_dir = Path("experiment_configs")
    
    if not configs_dir.exists():
        return []
    
    configs = []
    for yaml_file in configs_dir.glob("*.yaml"):
        configs.append(yaml_file.stem)
    
    return sorted(configs)


# Backwards compatibility aliases
MNISTConfig = lambda **kwargs: load_config(dataset='mnist', **kwargs)
CIFAR10Config = lambda **kwargs: load_config(dataset='cifar10', **kwargs)  
CIFAR100Config = lambda **kwargs: load_config(dataset='cifar100', **kwargs)
get_config = load_config