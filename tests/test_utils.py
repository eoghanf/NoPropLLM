"""
Tests for utility functions.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.utils import (
    set_seed, get_device, count_parameters, format_time,
    Timer, AverageMeter, save_checkpoint, load_checkpoint
)


class TestSeedAndDevice:
    """Test seed setting and device utilities."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        set_seed(42)
        tensor1 = torch.randn(5)
        
        set_seed(42)
        tensor2 = torch.randn(5)
        
        assert torch.allclose(tensor1, tensor2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']


class TestModelUtilities:
    """Test model-related utilities."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        # Simple model with known parameter count
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),  # 10*5 + 5 = 55 parameters
            torch.nn.Linear(5, 3)    # 5*3 + 3 = 18 parameters
        )
        # Total: 55 + 18 = 73 parameters
        
        param_count = count_parameters(model)
        assert param_count == 73
    
    def test_count_parameters_frozen(self):
        """Test parameter counting with frozen parameters."""
        model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        
        # Count all parameters
        all_params = count_parameters(model)
        assert all_params == 55
        
        # Freeze some parameters
        model.weight.requires_grad = False
        trainable_params = count_parameters(model)
        assert trainable_params == 5  # Only bias parameters


class TestTimeUtilities:
    """Test time-related utilities."""
    
    def test_format_time_seconds(self):
        """Test time formatting for seconds."""
        assert format_time(30.5) == "30.5s"
        assert format_time(45.0) == "45.0s"
    
    def test_format_time_minutes(self):
        """Test time formatting for minutes."""
        assert format_time(90.0) == "1m 30.0s"
        assert format_time(125.5) == "2m 5.5s"
    
    def test_format_time_hours(self):
        """Test time formatting for hours."""
        assert format_time(3661.0) == "1h 1m 1.0s"
        assert format_time(7200.0) == "2h 0m 0.0s"
    
    def test_timer(self):
        """Test Timer class."""
        timer = Timer()
        
        # Test error before starting
        with pytest.raises(RuntimeError):
            timer.elapsed()
        
        # Start timer
        timer.start()
        elapsed = timer.elapsed()
        assert elapsed >= 0
        
        # Stop timer
        import time
        time.sleep(0.01)  # Small sleep to ensure some time passes
        final_time = timer.stop()
        assert final_time > elapsed
        
        # Test elapsed after stop
        elapsed_after_stop = timer.elapsed()
        assert elapsed_after_stop == final_time


class TestAverageMeter:
    """Test AverageMeter class."""
    
    def test_average_meter_single_update(self):
        """Test AverageMeter with single update."""
        meter = AverageMeter()
        
        meter.update(5.0)
        assert meter.val == 5.0
        assert meter.avg == 5.0
        assert meter.sum == 5.0
        assert meter.count == 1
    
    def test_average_meter_multiple_updates(self):
        """Test AverageMeter with multiple updates."""
        meter = AverageMeter()
        
        meter.update(10.0, n=2)  # Add 10.0 twice
        meter.update(5.0, n=1)   # Add 5.0 once
        
        # Should have: (10*2 + 5*1) / (2+1) = 25/3 ≈ 8.333
        assert meter.count == 3
        assert meter.sum == 25.0
        assert abs(meter.avg - 25.0/3.0) < 1e-6
        assert meter.val == 5.0  # Last value
    
    def test_average_meter_reset(self):
        """Test AverageMeter reset functionality."""
        meter = AverageMeter()
        
        meter.update(10.0)
        assert meter.avg == 10.0
        
        meter.reset()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0


class TestCheckpointUtilities:
    """Test checkpoint saving and loading utilities."""
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        # Create a simple model
        model = torch.nn.Linear(5, 3)
        optimizers = {'opt1': torch.optim.Adam(model.parameters())}
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            save_checkpoint(
                model=model,
                optimizers=optimizers,
                epoch=10,
                best_accuracy=95.5,
                filepath=checkpoint_path,
                custom_data="test"
            )
            
            # Load checkpoint
            new_model = torch.nn.Linear(5, 3)
            new_optimizers = {'opt1': torch.optim.Adam(new_model.parameters())}
            
            checkpoint = load_checkpoint(
                filepath=checkpoint_path,
                model=new_model,
                optimizers=new_optimizers
            )
            
            # Verify loaded data
            assert checkpoint['epoch'] == 10
            assert checkpoint['best_accuracy'] == 95.5
            assert checkpoint['custom_data'] == "test"
            
            # Verify model state was loaded (parameters should match)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
                
        finally:
            Path(checkpoint_path).unlink()  # Clean up
    
    def test_checkpoint_load_without_optimizers(self):
        """Test loading checkpoint without optimizer state."""
        model = torch.nn.Linear(3, 2)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            save_checkpoint(
                model=model,
                optimizers={},
                epoch=5,
                best_accuracy=80.0,
                filepath=checkpoint_path
            )
            
            # Load without optimizers
            new_model = torch.nn.Linear(3, 2)
            checkpoint = load_checkpoint(checkpoint_path, new_model)
            
            assert checkpoint['epoch'] == 5
            assert checkpoint['best_accuracy'] == 80.0
            
        finally:
            Path(checkpoint_path).unlink()