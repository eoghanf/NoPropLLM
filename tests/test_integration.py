"""
Integration tests for end-to-end functionality.
These tests verify that the full training pipeline works correctly.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import NoPropTrainer, NoPropConfig, set_seed


class TestTrainingIntegration:
    """Integration tests for the training pipeline."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return NoPropConfig(
            dataset='mnist',
            epochs=2,  # Very short for testing
            batch_size=8,
            num_layers=3,  # Small network
            hidden_dim=32,
            learning_rate=0.01,
            seed=42,
            log_interval=1,  # Log every batch for testing
            save_best=False,  # Don't save during tests
            save_final=False
        )
    
    @pytest.fixture
    def fake_data(self):
        """Create fake data for testing."""
        # Create small fake dataset
        train_images = torch.randn(32, 1, 28, 28)
        train_labels = torch.randint(0, 10, (32,))
        test_images = torch.randn(16, 1, 28, 28)
        test_labels = torch.randint(0, 10, (16,))
        
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        return train_loader, test_loader
    
    @pytest.mark.integration
    def test_full_training_pipeline(self, simple_config, fake_data):
        """Test complete training pipeline with fake data."""
        set_seed(42)
        train_loader, test_loader = fake_data
        
        # Create trainer
        trainer = NoPropTrainer(simple_config)
        
        # Verify trainer setup
        assert trainer.model is not None
        assert len(trainer.optimizers) == simple_config.num_layers + 1  # layers + final
        assert trainer.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test initial validation (should not crash)
        initial_loss, initial_acc = trainer.validate(test_loader)
        assert initial_loss > 0
        assert 0 <= initial_acc <= 100
        
        # Test single training epoch (should not crash)
        train_loss = trainer.train_epoch(train_loader, epoch=1)
        assert train_loss >= 0
        
        # Test validation after training (should not crash)
        final_loss, final_acc = trainer.validate(test_loader)
        assert final_loss >= 0
        assert 0 <= final_acc <= 100
    
    @pytest.mark.integration  
    def test_noise_schedule_correctness(self, simple_config):
        """Test that noise schedule works correctly across layers."""
        set_seed(42)
        
        trainer = NoPropTrainer(simple_config)
        model = trainer.model
        
        batch_size = 4
        labels = torch.randint(0, 10, (batch_size,))
        
        # Test that different layers get different noise levels
        noise_levels = []
        for layer_idx in range(model.num_layers):
            noisy_label = model.get_noisy_label(labels, layer_idx)
            clean_embed = model.embed_matrix[labels]
            
            # Calculate noise level
            noise_ratio = torch.mean(torch.abs(noisy_label - clean_embed)).item()
            noise_levels.append(noise_ratio)
        
        # Verify decreasing noise schedule
        for i in range(len(noise_levels) - 1):
            assert noise_levels[i] >= noise_levels[i + 1], f"Noise should decrease: {noise_levels}"
    
    @pytest.mark.integration
    def test_loss_computation_all_layers(self, simple_config, fake_data):
        """Test that loss computation works for all layers."""
        set_seed(42)
        train_loader, _ = fake_data
        
        trainer = NoPropTrainer(simple_config)
        model = trainer.model
        
        # Get a batch of data
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images = images.to(trainer.device)
        labels = labels.to(trainer.device)
        
        # Test loss computation for each layer
        for layer_idx in range(model.num_layers):
            loss = model.compute_loss(images, labels, layer_idx)
            
            assert loss.item() >= 0, f"Loss should be non-negative for layer {layer_idx}"
            assert not torch.isnan(loss), f"Loss should not be NaN for layer {layer_idx}"
            assert loss.requires_grad, f"Loss should require gradients for layer {layer_idx}"
        
        # Test classifier loss
        classifier_loss = model.compute_classifier_loss(images, labels)
        assert classifier_loss.item() >= 0
        assert not torch.isnan(classifier_loss)
        assert classifier_loss.requires_grad
    
    @pytest.mark.integration
    def test_gradient_flow_independence(self, simple_config, fake_data):
        """Test that gradients flow independently for each layer."""
        set_seed(42)
        train_loader, _ = fake_data
        
        trainer = NoPropTrainer(simple_config)
        model = trainer.model
        
        # Get a batch of data
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images = images.to(trainer.device)
        labels = labels.to(trainer.device)
        
        # Test gradient flow for first layer
        model.zero_grad()
        loss_0 = model.compute_loss(images, labels, 0)
        loss_0.backward()
        
        # Check that only first layer has gradients
        first_layer_has_grads = any(p.grad is not None for p in model.denoising_modules[0].parameters())
        other_layers_no_grads = all(
            all(p.grad is None for p in model.denoising_modules[i].parameters())
            for i in range(1, model.num_layers)
        )
        
        assert first_layer_has_grads, "First layer should have gradients"
        assert other_layers_no_grads, "Other layers should not have gradients"
    
    @pytest.mark.integration
    def test_inference_mode(self, simple_config, fake_data):
        """Test that inference mode works correctly."""
        set_seed(42)
        _, test_loader = fake_data
        
        trainer = NoPropTrainer(simple_config)
        model = trainer.model
        
        # Test inference
        model.eval()
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images = images.to(trainer.device)
        
        with torch.no_grad():
            output = model(images, mode='inference')
            
            # Should be log probabilities
            assert output.shape == (images.shape[0], 10)
            
            # Convert to probabilities and check they sum to 1
            probs = torch.exp(output)
            prob_sums = probs.sum(dim=1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_minimal_training_run(self, simple_config, fake_data):
        """Test a minimal but complete training run."""
        set_seed(42)
        train_loader, test_loader = fake_data
        
        # Use even shorter config for this test
        config = simple_config.update(epochs=1, log_interval=10)
        
        trainer = NoPropTrainer(config)
        
        # Run training (should not crash)
        trainer.train(train_loader, test_loader)
        
        # Verify training completed
        assert trainer.current_epoch == 1
        assert len(trainer.training_history) == 1
        
        # Verify history contains expected keys
        history_entry = trainer.training_history[0]
        expected_keys = {'epoch', 'train_loss', 'test_loss', 'accuracy', 'time'}
        assert set(history_entry.keys()) >= expected_keys


@pytest.fixture(autouse=True)
def cleanup_after_integration_tests():
    """Clean up any files created during integration tests."""
    yield
    
    # Clean up any model files that might have been created
    import os
    for filename in ['best_model.pt', 'final_model.pt', 'test_model.pt']:
        if os.path.exists(filename):
            os.remove(filename)