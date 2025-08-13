"""
Tests for NoProp models.
"""

import pytest
import torch
import numpy as np

from src.models import DenoisingModule, NoPropNetwork


class TestDenoisingModule:
    """Test DenoisingModule class."""
    
    @pytest.fixture
    def denoising_module(self):
        """Create a test denoising module."""
        return DenoisingModule(
            image_channels=1,
            image_size=28,
            label_dim=10
        )
    
    def test_initialization(self, denoising_module):
        """Test module initialization."""
        assert denoising_module.image_channels == 1
        assert denoising_module.image_size == 28
        assert denoising_module.label_dim == 10
    
    def test_forward_pass(self, denoising_module):
        """Test forward pass through denoising module."""
        batch_size = 4
        image = torch.randn(batch_size, 1, 28, 28)
        noised_label = torch.randn(batch_size, 10)
        
        # Test logits output
        output = denoising_module(image, noised_label)
        assert output.shape == (batch_size, 10)
    
    def test_different_image_sizes(self):
        """Test module with different image sizes."""
        # MNIST (28x28)
        mnist_module = DenoisingModule(image_channels=1, image_size=28, label_dim=10)
        mnist_input = torch.randn(2, 1, 28, 28)
        mnist_label = torch.randn(2, 10)
        mnist_output = mnist_module(mnist_input, mnist_label)
        assert mnist_output.shape == (2, 10)
        
        # CIFAR (32x32)
        cifar_module = DenoisingModule(image_channels=3, image_size=32, label_dim=10)
        cifar_input = torch.randn(2, 3, 32, 32)
        cifar_label = torch.randn(2, 10)
        cifar_output = cifar_module(cifar_input, cifar_label)
        assert cifar_output.shape == (2, 10)


class TestNoPropNetwork:
    """Test NoPropNetwork class."""
    
    @pytest.fixture
    def noprop_network(self):
        """Create a test NoProp network."""
        return NoPropNetwork(
            num_layers=5,  # Smaller for faster tests
            image_channels=1,
            image_size=28,
            label_dim=10
        )
    
    def test_initialization(self, noprop_network):
        """Test network initialization."""
        assert noprop_network.num_layers == 5
        assert noprop_network.label_dim == 10
        assert len(noprop_network.denoising_modules) == 5
        assert noprop_network.alphas.shape == (5,)
    
    def test_noise_schedule(self, noprop_network):
        """Test noise level generation for layers."""
        batch_size = 4
        labels = torch.randint(0, 10, (batch_size,))
        
        # Test different noise levels for different layers
        noise_levels = []
        for layer_idx in range(noprop_network.num_layers):
            noisy_label = noprop_network.get_noisy_label(labels, layer_idx)
            
            # Calculate approximate noise level
            clean_embed = torch.zeros(batch_size, 10)
            clean_embed.scatter_(1, labels.unsqueeze(1), 1.0)
            noise_ratio = torch.mean(torch.abs(noisy_label - clean_embed)).item()
            noise_levels.append(noise_ratio)
        
        # Early layers should have more noise than later layers
        assert noise_levels[0] > noise_levels[-1]
    
    def test_forward_training(self, noprop_network):
        """Test forward pass during training."""
        batch_size = 4
        image = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,))
        
        for layer_idx in range(noprop_network.num_layers):
            prediction = noprop_network.forward_training(image, labels, layer_idx)
            assert prediction.shape == (batch_size, 10)
    
    def test_forward_inference(self, noprop_network):
        """Test forward pass during inference."""
        batch_size = 4
        image = torch.randn(batch_size, 1, 28, 28)
        
        output = noprop_network.forward_inference(image)
        assert output.shape == (batch_size, 10)
        
        # Should be log probabilities (negative values, sum to ~0 in probability space)
        probs = torch.exp(output)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    def test_compute_loss(self, noprop_network):
        """Test loss computation for different layers."""
        batch_size = 4
        image = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,))
        
        losses = []
        for layer_idx in range(noprop_network.num_layers):
            loss = noprop_network.compute_loss(image, labels, layer_idx)
            assert loss.item() >= 0  # Loss should be non-negative
            assert not torch.isnan(loss)  # Loss should not be NaN
            losses.append(loss.item())
        
        # All losses should be different (due to different noise levels)
        assert len(set(np.round(losses, 4))) > 1
    
    def test_compute_classifier_loss(self, noprop_network):
        """Test classifier loss computation."""
        batch_size = 4
        image = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,))
        
        classifier_loss = noprop_network.compute_classifier_loss(image, labels)
        assert classifier_loss.item() >= 0
        assert not torch.isnan(classifier_loss)
    
    def test_different_architectures(self):
        """Test networks with different architectures."""
        # Test CIFAR-like network
        cifar_net = NoPropNetwork(
            num_layers=3,
            image_channels=3,
            image_size=32,
            label_dim=100
        )
        
        batch_size = 2
        image = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 100, (batch_size,))
        
        # Test inference
        output = cifar_net.forward_inference(image)
        assert output.shape == (batch_size, 100)
        
        # Test training
        loss = cifar_net.compute_loss(image, labels, 0)
        assert loss.item() >= 0
    
    def test_gradient_flow(self, noprop_network):
        """Test that gradients flow properly through the network."""
        batch_size = 2
        image = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Test gradient flow for layer training
        loss = noprop_network.compute_loss(image, labels, 0)
        loss.backward()
        
        # Check that gradients exist for the first denoising module
        for param in noprop_network.denoising_modules[0].parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
        
        # Clear gradients and test classifier loss
        noprop_network.zero_grad()
        classifier_loss = noprop_network.compute_classifier_loss(image, labels)
        classifier_loss.backward()
        
        # Check gradients for classifier
        for param in noprop_network.classifier.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()