import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for interpreting neural network activations.
    
    Args:
        input_dim (int): Dimension of input activations
        hidden_dim (int): Dimension of the sparse representation (typically > input_dim)
        l1_coefficient (float): Coefficient for L1 sparsity penalty
        use_top_k (bool): Whether to use top-k sparsity in the encoder
        top_k_percentage (float): Percentage of features to keep active (between 0 and 1)
        dtype (torch.dtype): Data type for parameters
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        l1_coefficient: float = 1e-3,
        use_top_k: bool = False,
        top_k_percentage: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coefficient = l1_coefficient
        self.use_top_k = use_top_k
        self.top_k_percentage = top_k_percentage
        
        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True, dtype=dtype)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='linear')
        nn.init.zeros_(self.decoder.bias)
    
    def _apply_top_k(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k sparsity to each sample in the batch.
        
        Args:
            x: Tensor of shape [batch_size, hidden_dim]
            
        Returns:
            Tensor with only top-k values preserved
        """
        batch_size = x.shape[0]
        k = max(1, int(self.top_k_percentage * self.hidden_dim))
        
        # Find threshold value for each sample in batch
        values, _ = torch.topk(x, k, dim=1)
        thresholds = values[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Zero out values below the threshold
        return x * (x >= thresholds)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_input, sparse_code)
        """
        # Encode to get sparse features
        sparse_code = F.relu(self.encoder(x))
        
        # Apply top-k sparsity if enabled
        if self.use_top_k:
            sparse_code = self._apply_top_k(sparse_code)
        
        # Decode to reconstruct input
        reconstructed = self.decoder(sparse_code)
        
        return reconstructed, sparse_code
    
    def loss(
        self, 
        x: torch.Tensor, 
        reconstructed: torch.Tensor, 
        sparse_code: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute autoencoder loss: reconstruction loss + sparsity penalty.
        
        Args:
            x: Original input
            reconstructed: Reconstructed input
            sparse_code: Sparse code from encoder
            
        Returns:
            Dictionary containing loss components
        """
        # MSE reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # L1 sparsity loss
        l1_loss = torch.tensor(0.0, device=x.device)
        if self.l1_coefficient > 0:
            l1_loss = self.l1_coefficient * sparse_code.abs().mean()
        
        # Total loss
        total_loss = recon_loss + l1_loss
        
        # Calculate sparsity metrics
        sparsity = (sparse_code == 0).float().mean().item()
        mean_activation = sparse_code.abs().mean().item()
        
        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "l1_sparsity": l1_loss,
            "mean_activation": mean_activation,
            "sparsity": sparsity,
        }
        
    def get_feature_importance(self) -> torch.Tensor:
        """
        Get importance of each feature in the sparse representation.
        
        Returns:
            Tensor with importance score for each feature
        """
        # L2 norm of the decoder weights for each feature
        return torch.norm(self.decoder.weight, dim=0)
    
    def get_top_activating_inputs(
        self, 
        inputs: torch.Tensor, 
        feature_idx: int, 
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find inputs that most strongly activate a specific feature.
        
        Args:
            inputs: Batch of inputs to check [batch_size, input_dim]
            feature_idx: Index of feature to analyze
            top_k: Number of top activating inputs to return
            
        Returns:
            Tuple of (top_k_inputs, activation_values)
        """
        with torch.no_grad():
            _, sparse_codes = self.forward(inputs)
            activations = sparse_codes[:, feature_idx]
            
            # Get top-k activating indices
            top_k_values, top_k_indices = torch.topk(activations, min(top_k, len(activations)))
            top_k_inputs = inputs[top_k_indices]
            
            return top_k_inputs, top_k_values 