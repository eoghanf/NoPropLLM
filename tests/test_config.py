"""
Tests for configuration management.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import NoPropConfig, load_config, list_available_configs


class TestNoPropConfig:
    """Test NoPropConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = NoPropConfig()
        
        assert config.dataset == "mnist"
        assert config.batch_size == 128
        assert config.epochs == 100
        assert config.num_layers == 10
    
    def test_yaml_loading(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        test_config = {
            'dataset': 'test_dataset',
            'batch_size': 64,
            'epochs': 50,
            'learning_rate': 0.01
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(test_config, f)
            temp_path = f.name
        
        try:
            config = NoPropConfig.from_yaml(temp_path)
            
            assert config.dataset == 'test_dataset'
            assert config.batch_size == 64
            assert config.epochs == 50
            assert config.learning_rate == 0.01
            assert config._config_path == temp_path
        finally:
            Path(temp_path).unlink()  # Clean up
    
    def test_config_update(self):
        """Test configuration update functionality."""
        config = NoPropConfig(dataset='mnist', epochs=100)
        updated_config = config.update(epochs=200, batch_size=256)
        
        # Original config unchanged
        assert config.epochs == 100
        assert config.batch_size == 128
        
        # Updated config has new values
        assert updated_config.epochs == 200
        assert updated_config.batch_size == 256
        assert updated_config.dataset == 'mnist'  # Unchanged value preserved
    
    def test_yaml_save(self):
        """Test saving configuration to YAML file."""
        config = NoPropConfig(dataset='test', epochs=123, batch_size=456)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_yaml(temp_path)
            
            # Load back and verify
            with open(temp_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config['dataset'] == 'test'
            assert saved_config['epochs'] == 123
            assert saved_config['batch_size'] == 456
        finally:
            Path(temp_path).unlink()


class TestConfigLoading:
    """Test config loading utilities."""
    
    def test_load_config_with_dataset(self):
        """Test loading config by dataset name."""
        if Path("experiment_configs/mnist.yaml").exists():
            config = load_config(dataset='mnist')
            assert config.dataset == 'mnist'
            assert config._config_path is not None
    
    def test_load_config_with_overrides(self):
        """Test loading config with parameter overrides."""
        if Path("experiment_configs/mnist.yaml").exists():
            config = load_config(dataset='mnist', epochs=999, batch_size=32)
            assert config.epochs == 999
            assert config.batch_size == 32
    
    def test_load_config_missing_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config(dataset='nonexistent_dataset')
    
    def test_load_config_no_args(self):
        """Test error when no config source provided."""
        with pytest.raises(ValueError, match="Either config_path or dataset must be provided"):
            load_config()


class TestConfigUtilities:
    """Test utility functions."""
    
    def test_list_available_configs(self):
        """Test listing available configuration files."""
        configs = list_available_configs()
        assert isinstance(configs, list)
        
        # If experiment_configs exists and has files, should find them
        configs_dir = Path("experiment_configs")
        if configs_dir.exists():
            yaml_files = list(configs_dir.glob("*.yaml"))
            expected_configs = sorted([f.stem for f in yaml_files])
            assert configs == expected_configs