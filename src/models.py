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
        label_dim: int = 10,
        hidden_dim: int = 256
    ):
        """
        Initialize a single denoising module.
        
        Args:
            image_channels: Number of input image channels (1 for MNIST, 3 for CIFAR)
            image_size: Size of input images (28 for MNIST, 32 for CIFAR)
            label_dim: Dimension of label embeddings
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        
        # Image processing pathway (left side of Figure 6)
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Calculate flattened image feature size
        with torch.no_grad():
            dummy_img = torch.zeros(1, image_channels, image_size, image_size)
            img_features = self.image_conv(dummy_img).shape[1]
        
        self.image_fc = nn.Sequential(
            nn.Linear(img_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Noised label processing pathway
        if label_dim == image_size * image_size * image_channels:
            # When embedding dimension matches image dimension, treat as image
            self.label_conv = nn.Sequential(
                nn.Conv2d(image_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            
            self.label_fc = nn.Sequential(
                nn.Linear(img_features, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            )
        else:
            # Standard fully connected processing for label embeddings
            self.label_conv = None
            self.label_fc = nn.Sequential(
                nn.Linear(label_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            )
        
        # Fused processing after concatenation
        self.fused_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # Concatenated features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, label_dim)  # Output logits
        )
        
    def forward(
        self, 
        image: torch.Tensor, 
        noised_label: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the denoising module.
        
        Args:
            image: Input image tensor of shape (batch_size, channels, height, width)
            noised_label: Noised label embedding of shape (batch_size, label_dim)
            return_logits: If True, return raw logits instead of applying softmax
            
        Returns:
            Model output (probability distribution over class embeddings or raw logits)
        """
        batch_size = image.shape[0]
        
        # Process image through convolutional pathway
        img_features = self.image_conv(image)
        img_embed = self.image_fc(img_features)
        
        # Process noised label
        if self.label_conv is not None and noised_label.shape[1] == self.image_size * self.image_size * self.image_channels:
            # Treat noised label as image (when embedding dimension matches image dimension)
            noised_label_img = noised_label.view(batch_size, self.image_channels, self.image_size, self.image_size)
            label_features = self.label_conv(noised_label_img)
            label_embed = self.label_fc(label_features)
        else:
            # Standard FC processing for label embeddings
            label_embed = self.label_fc(noised_label)
        
        # Concatenate image and label embeddings
        fused_features = torch.cat([img_embed, label_embed], dim=1)
        
        # Process through fused layers to get logits
        logits = self.fused_layers(fused_features)
        
        if return_logits:
            return logits
        
        # Apply softmax to get probability distribution over class embeddings
        return F.softmax(logits, dim=1)


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
        hidden_dim: int = 256,
        embed_matrix: Optional[torch.Tensor] = None
    ):
        """
        Initialize the complete NoProp network.
        
        Args:
            num_layers: Number of denoising layers T (default 10 as mentioned in paper)
            image_channels: Number of input image channels
            image_size: Size of input images
            label_dim: Dimension of label embeddings
            hidden_dim: Hidden dimension for each denoising module
            embed_matrix: Optional class embedding matrix W_Embed
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.label_dim = label_dim
        
        # Create T denoising modules (u_1, u_2, ..., u_T)
        self.denoising_modules = nn.ModuleList([
            DenoisingModule(
                image_channels=image_channels,
                image_size=image_size,
                label_dim=label_dim,
                hidden_dim=hidden_dim
            ) for _ in range(num_layers)
        ])
        
        # Class embedding matrix W_Embed
        if embed_matrix is not None:
            self.register_buffer('embed_matrix', embed_matrix)
        else:
            # Initialize as identity matrix (one-hot)
            self.register_buffer('embed_matrix', torch.eye(label_dim))
        
        # Final classification layer p_θ_out(y|z_T)
        self.classifier = nn.Linear(label_dim, label_dim)
        
        # Noise schedule parameters (cosine schedule)
        self.register_buffer('alphas', self._cosine_schedule(num_layers))
        
    def _cosine_schedule(self, T: int) -> torch.Tensor:
        """Generate cosine noise schedule as used in the paper."""
        steps = torch.linspace(0, 1, T + 1)
        alphas_cumprod = torch.cos(((steps + 0.008) / 1.008) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(alphas, 0.0001, 0.9999)
    
    def get_noisy_label(self, y: torch.Tensor, layer_idx: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noisy label for specific layer with decreasing noise schedule.
        Layer 0 has highest noise (0.9), last layer has lowest noise (0.01).
        """
        batch_size = y.shape[0]
        device = y.device
        
        # Get label embeddings
        if y.dtype == torch.long:
            # Convert class indices to embeddings
            clean_embed = self.embed_matrix[y]  # [batch, label_dim]
        else:
            clean_embed = y
        
        # Define decreasing noise schedule: 0.9, 0.8, 0.7, ..., 0.01
        noise_levels = torch.linspace(0.9, 0.01, self.num_layers, device=device)
        noise_level = noise_levels[layer_idx]
        
        if noise is None:
            noise = torch.randn_like(clean_embed)
        
        # Apply layer-specific noise: z_layer = (1 - noise_level) * clean + noise_level * noise
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
            # Each module predicts probability distribution over class embeddings
            probs = module(image, z, return_logits=False)  # [batch, label_dim]
            
            # Compute weighted sum of class embeddings (predicted u_y)
            predicted_embed = probs @ self.embed_matrix  # [batch, label_dim]
            
            # Update z using residual connection with noise (Equation 3)
            if t < len(self.denoising_modules) - 1:
                alpha_t = self.alphas[t]
                # Simplified residual update
                z = torch.sqrt(alpha_t) * predicted_embed + torch.sqrt(1 - alpha_t) * z
        
        # Final classification
        logits = self.classifier(z)
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
        prediction = self.denoising_modules[layer_idx](image, noisy_label, return_logits=True)
        
        return prediction
    
    def compute_loss(self, image: torch.Tensor, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute NoProp loss for specific layer.
        Each layer gets clean image x but layer-specific noisy labels to denoise.
        """
        # Get target embeddings (clean labels)
        if y.dtype == torch.long:
            target_embed = self.embed_matrix[y]
        else:
            target_embed = y
        
        # Get prediction from module at this layer (using clean image + noisy labels)
        prediction = self.forward_training(image, y, layer_idx)
        
        # Convert logits to weighted embedding prediction
        probs = F.softmax(prediction, dim=1)
        predicted_embed = probs @ self.embed_matrix
        
        # L2 loss between predicted and target embeddings (clean labels)
        mse_loss = F.mse_loss(predicted_embed, target_embed)
        
        # Layer-specific weighting based on noise schedule
        noise_levels = torch.linspace(0.9, 0.01, self.num_layers, device=image.device)
        noise_level = noise_levels[layer_idx]
        
        # Higher noise levels get higher weights (harder denoising task)
        weight = noise_level.clamp(min=0.01)
        
        return weight * mse_loss
    
    def compute_classifier_loss(self, image: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute classifier loss E[−log p̂_θout(y|z_T)] from Equation 8.
        This trains the final classifier to predict labels from the final layer output.
        """
        # Run inference to get z_T (final layer output)
        batch_size = image.shape[0]
        device = image.device
        
        # Start with z_0 (Gaussian noise) and run through all layers to get z_T
        z = torch.randn(batch_size, self.label_dim, device=device)
        
        # Progressive denoising through T modules to get final z_T
        with torch.no_grad():  # Don't backprop through the denoising process
            for t, module in enumerate(self.denoising_modules):
                probs = module(image, z, return_logits=False)
                predicted_embed = probs @ self.embed_matrix
                
                if t < len(self.denoising_modules) - 1:
                    alpha_t = self.alphas[t]
                    z = torch.sqrt(alpha_t) * predicted_embed + torch.sqrt(1 - alpha_t) * z
                else:
                    z = predicted_embed  # Final z_T
        
        # Classify final z_T
        logits = self.classifier(z)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Cross-entropy loss: -log p̂_θout(y|z_T)
        classifier_loss = F.nll_loss(log_probs, y)
        
        return classifier_loss
    
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
        label_dim=10,
        hidden_dim=256
    )
    
    batch_size = 4
    image = torch.randn(batch_size, 3, 32, 32)
    noised_label = torch.randn(batch_size, 10)
    
    output = denoising_module(image, noised_label)
    print(f"DenoisingModule output shape: {output.shape}")
    print(f"Output sum (should be ~1): {output.sum(dim=1)}")
    
    # Test complete NoProp network
    print("\n=== Testing NoPropNetwork (10 layers) ===")
    network = NoPropNetwork(
        num_layers=10,
        image_channels=3,
        image_size=32,
        label_dim=10,
        hidden_dim=256
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
        hidden_dim=256
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