import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DenoisingModule(nn.Module):
    """
    Single denoising module (u_t) from Figure 6 of the NoProp paper.
    
    Each module learns to denoise a noisy target by taking image x and noised label z_{t-1}
    and predicting the clean label embedding u_y.
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 32,
        label_dim: int = 10
    ):
        """
        Initialize a single denoising module.
        
        Args:
            image_channels: Number of input image channels (1 for MNIST, 3 for CIFAR)
            image_size: Size of input images (28 for MNIST, 32 for CIFAR)
            label_dim: Dimension of label embeddings (same as number of classes)
        """
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.label_dim = label_dim
        
        # Image processing pathway (left side of Figure 6)
        # Calculate flattened size after conv layers: image_size -> /2 -> /2 = image_size/4
        conv_output_size = (image_size // 4) ** 2 * 64  # 64 channels from final conv layer
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(in_features=conv_output_size, out_features=256),
            nn.BatchNorm1d(256)
        )

        self.noised_label_b1= nn.Sequential(
            nn.Linear(label_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.noised_label_b2 = nn.Sequential(
            nn.Linear(in_features=256,
                      out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(256),
            nn.Linear(in_features=256,
                      out_features=256),
            nn.BatchNorm1d(256)
        )
        # Fused processing after concatenation
        self.fused_layers = nn.Sequential(
            nn.Linear(256 + 256, 256),  # img_embed (256) + label_embed (256)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, label_dim)  # Output logits
        )
        
    def forward(
        self, 
        image: torch.Tensor, 
        noised_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the denoising module.
        
        Args:
            image: Input image tensor of shape (batch_size, channels, height, width)
            noised_label: Noised label embedding of shape (batch_size, label_dim)
            
        Returns:
            Model output logits of shape (batch_size, label_dim)
        """
        batch_size = image.shape[0]
        
        # Process image through convolutional pathway
        img_embed = self.image_conv(image)
        
        label_embed_s1 = self.noised_label_b1(noised_label)
        label_embed = self.noised_label_b2(label_embed_s1) + label_embed_s1
        
        # Concatenate image and label embeddings
        fused_features = torch.cat([img_embed, label_embed], dim=1)
        
        # Process through fused layers to get logits
        logits = self.fused_layers(fused_features)
        
        return logits


class NoPropNetwork(nn.Module):
    """
    Complete NoProp network with T denoising modules as shown in Figure 1.
    
    The network consists of T stacked DenoisingModules, each learning independently
    to denoise noisy labels, with a final classification layer.
    """
    
    def __init__(
        self,
        num_layers: int = 10,
        image_channels: int = 3,
        image_size: int = 32,
        label_dim: int = 10,
        noise_schedule_type: str = "cosine",
        noise_schedule_min: float = 0.001,
        noise_schedule_max: float = 0.999
    ):
        """
        Initialize the complete NoProp network.
        
        Args:
            num_layers: Number of denoising layers T (default 10 as mentioned in paper)
            image_channels: Number of input image channels
            image_size: Size of input images
            label_dim: Dimension of label embeddings (same as number of classes)
            noise_schedule_type: Type of noise schedule ("cosine" or "linear")
            noise_schedule_min: Minimum noise level
            noise_schedule_max: Maximum noise level
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.label_dim = label_dim
        self.noise_schedule_type = noise_schedule_type
        self.noise_schedule_min = noise_schedule_min
        self.noise_schedule_max = noise_schedule_max
        
        # Create T denoising modules (u_1, u_2, ..., u_T)
        self.denoising_modules = nn.ModuleList([
            DenoisingModule(
                image_channels=image_channels,
                image_size=image_size,
                label_dim=label_dim
            ) for _ in range(num_layers)
        ])
        
        
        # Final classification layer - identity function (z_T directly becomes prediction)
        self.classifier = nn.Identity()
        
        # Noise schedule parameters
        self.register_buffer('alphas', self._get_alpha_schedule(num_layers, noise_schedule_type))
        self.register_buffer('noise_schedule', self._get_noise_schedule(num_layers, noise_schedule_type, noise_schedule_min, noise_schedule_max))
        
    def _get_alpha_schedule(self, T: int, schedule_type: str) -> torch.Tensor:
        """Generate alpha schedule for inference denoising based on schedule type."""
        if schedule_type == "cosine":
            return self._cosine_alpha_schedule(T)
        elif schedule_type == "linear":
            return self._linear_alpha_schedule(T)
        else:
            raise ValueError(f"Unknown noise schedule type: {schedule_type}. Use 'cosine' or 'linear'.")
    
    def _cosine_alpha_schedule(self, T: int) -> torch.Tensor:
        """Generate cosine alpha schedule as used in the NoProp paper."""
        steps = torch.linspace(0, 1, T + 1)
        alphas_cumprod = torch.cos(((steps + 0.008) / 1.008) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(alphas, 0.0001, 0.9999)
    
    def _linear_alpha_schedule(self, T: int) -> torch.Tensor:
        """Generate linear alpha schedule for inference denoising."""
        # Linear schedule for alphas: start high (close to 1) and decrease linearly
        alphas = torch.linspace(0.9999, 0.0001, T)
        return alphas
    
    def _get_noise_schedule(self, T: int, schedule_type: str, min_noise: float, max_noise: float) -> torch.Tensor:
        """
        Generate noise schedule for training.
        
        Args:
            T: Number of layers
            schedule_type: "cosine" or "linear"
            min_noise: Minimum noise level (final layer)
            max_noise: Maximum noise level (first layer)
        """
        if schedule_type == "cosine":
            # Cosine schedule: starts at high noise, decreases to low noise
            steps = torch.linspace(0, 1, T)
            cosine_schedule = torch.cos(steps * torch.pi / 2) ** 2
            # Rescale to desired range: [max_noise, min_noise]
            noise_schedule = max_noise * cosine_schedule + min_noise * (1 - cosine_schedule)
            return noise_schedule
        elif schedule_type == "linear":
            # Linear schedule: linearly decrease from max to min noise
            return torch.linspace(max_noise, min_noise, T)
        else:
            raise ValueError(f"Unknown noise schedule type: {schedule_type}. Use 'cosine' or 'linear'.")
    
    def get_noisy_label(self, y: torch.Tensor, layer_idx: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noisy label for specific layer using cosine noise schedule.
        Layer 0 has highest noise, last layer has lowest noise (cosine decay).
        """
        batch_size = y.shape[0]
        device = y.device
        
        # Convert class indices to one-hot embeddings
        if y.dtype == torch.long:
            clean_embed = torch.zeros(batch_size, self.label_dim, device=device)
            clean_embed.scatter_(1, y.unsqueeze(1), 1.0)  # One-hot encoding
        else:
            clean_embed = y
        
        # Get noise level from cosine schedule
        noise_level = self.noise_schedule[layer_idx].to(device)
        
        if noise is None:
            noise = torch.randn_like(clean_embed)
        
        # Apply layer-specific noise: z_layer = sqrt(1 - noise_level) * clean + sqrt(noise_level) * noise
        noisy_label = torch.sqrt(1 - noise_level) * clean_embed + torch.sqrt(noise_level) * noise
        
        return noisy_label
    
    
    def forward_inference(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during inference (Figure 1 setup).
        Starts with z_0 (noise) and progressively denoises through T modules.
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Start with z_0 (Gaussian noise)
        z = torch.randn(batch_size, self.label_dim, device=device)
        
        # Progressive denoising through T modules
        for t, module in enumerate(self.denoising_modules):
            # Each module predicts logits over classes
            logits = module(image, z)  # [batch, label_dim]
            
            # Convert to probabilities (predicted clean labels)
            predicted_probs = torch.softmax(logits, dim=1)  # [batch, label_dim]
            
            # Update z using residual connection with noise (Equation 3)
            if t < len(self.denoising_modules) - 1:
                alpha_t = self.alphas[t]
                # Simplified residual update
                z = torch.sqrt(alpha_t) * predicted_probs + torch.sqrt(1 - alpha_t) * z
        
        # Final classification (identity - z_T directly becomes prediction)
        logits = self.classifier(z)  # Identity function: returns z unchanged
        return F.log_softmax(logits, dim=1)
    
    def forward_training(self, image: torch.Tensor, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass during training for a specific layer.
        Each layer gets the CLEAN image x but different noise levels on labels only.
        This matches Figure 1 where x goes to each u_t without modification.
        """
        # Generate layer-specific noisy labels only (image stays clean)
        noisy_label = self.get_noisy_label(y, layer_idx)
        
        # Get prediction from denoising module at this layer (clean image + noisy label)
        prediction = self.denoising_modules[layer_idx](image, noisy_label)
        
        return prediction
    
    def compute_loss(self, image: torch.Tensor, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute NoProp loss for specific layer.
        Each layer gets clean image x but layer-specific noisy labels to denoise.
        """
        # Get target one-hot embeddings (clean labels)
        batch_size = y.shape[0]
        device = y.device
        
        if y.dtype == torch.long:
            target_embed = torch.zeros(batch_size, self.label_dim, device=device)
            target_embed.scatter_(1, y.unsqueeze(1), 1.0)  # One-hot encoding
        else:
            target_embed = y
        
        # Get prediction from module at this layer (using clean image + noisy labels)
        prediction = self.forward_training(image, y, layer_idx)
        
        # Convert logits to probabilities (predicted clean labels)
        predicted_probs = F.softmax(prediction, dim=1)
        
        # L2 loss between predicted probabilities and target one-hot
        cross_entropy_loss = F.mse_loss(predicted_probs, target_embed)
        
        # Layer-specific weighting based on cosine noise schedule
        noise_level = self.noise_schedule[layer_idx].to(device)
        
        # Higher noise levels get higher weights (harder denoising task)
        weight = noise_level.clamp(min=0.01)
        
        return cross_entropy_loss
    
    def compute_classifier_loss(self, image: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute classifier loss. With identity classifier, this is equivalent to the final layer's denoising loss.
        The final denoising layer output directly becomes the classification prediction.
        """
        # With identity classifier, classification loss = final layer denoising loss
        final_layer_idx = self.num_layers - 1
        return self.compute_loss(image, y, final_layer_idx)
    
    def forward(self, image: torch.Tensor, y: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> torch.Tensor:
        """
        Forward pass - supports both inference and training modes.
        
        Args:
            image: Input images
            y: Target labels (required for training)
            mode: 'inference' or 'training'
        """
        if mode == 'inference':
            return self.forward_inference(image)
        elif mode == 'training':
            if y is None:
                raise ValueError("Target labels y required for training mode")
            # For training, we typically train one module at a time
            # This is handled by compute_loss method
            return self.forward_inference(image)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing NoProp implementation...")
    
    # Test single denoising module
    print("\n=== Testing DenoisingModule ===")
    denoising_module = DenoisingModule(
        image_channels=3,
        image_size=32,
        label_dim=10
    )
    
    batch_size = 4
    image = torch.randn(batch_size, 3, 32, 32)
    noised_label = torch.randn(batch_size, 10)
    
    output = denoising_module(image, noised_label)
    print(f"DenoisingModule output shape: {output.shape}")
    print(f"Output (logits): {output[0][:3]} ...")  # Show first 3 logits of first sample
    
    # Test complete NoProp network
    print("\n=== Testing NoPropNetwork (10 layers) ===")
    network = NoPropNetwork(
        num_layers=10,
        image_channels=3,
        image_size=32,
        label_dim=10,
        noise_schedule_type="cosine"
    )
    
    # Test inference
    print("Testing inference mode...")
    inference_output = network(image, mode='inference')
    print(f"Inference output shape: {inference_output.shape}")
    
    # Test training loss computation
    print("Testing training loss computation...")
    labels = torch.randint(0, 10, (batch_size,))
    
    total_loss = 0
    for t in range(network.num_layers):
        loss_t = network.compute_loss(image, labels, t)
        total_loss += loss_t
        print(f"Loss at time step {t}: {loss_t.item():.4f}")
    
    print(f"Total training loss: {total_loss.item():.4f}")
    
    # Test MNIST configuration
    print("\n=== Testing MNIST configuration ===")
    mnist_network = NoPropNetwork(
        num_layers=10,
        image_channels=1,
        image_size=28,
        label_dim=10,
        noise_schedule_type="cosine"
    )
    
    mnist_image = torch.randn(batch_size, 1, 28, 28)
    mnist_output = mnist_network(mnist_image, mode='inference')
    print(f"MNIST inference output shape: {mnist_output.shape}")
    
    # Print parameter counts
    denoising_params = sum(p.numel() for p in denoising_module.parameters())
    network_params = sum(p.numel() for p in network.parameters())
    
    print(f"\n=== Parameter Counts ===")
    print(f"Single DenoisingModule parameters: {denoising_params:,}")
    print(f"Complete NoPropNetwork (10 layers): {network_params:,}")
    print(f"Parameters per layer: {network_params // 10:,} (approx)")
    
    print("\nAll tests passed!")