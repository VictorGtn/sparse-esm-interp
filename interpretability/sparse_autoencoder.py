import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import deque


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
        data_for_init (torch.Tensor, optional): Data to compute geometric median for bias initialization
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        l1_coefficient: float = 1e-3,
        use_top_k: bool = False,
        top_k_percentage: float = 0.1,
        dtype: torch.dtype = torch.float32,
        data_for_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coefficient = l1_coefficient
        self.use_top_k = use_top_k
        self.top_k_percentage = top_k_percentage
        
        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False, dtype=dtype)
        
        # Learned centering bias (shared between encoder and decoder)
        # We will constrain pre-encoder bias = -post-decoder bias
        self.centering_bias = nn.Parameter(torch.zeros(input_dim, dtype=dtype))
        
        # Initialize weights
        self._init_weights(data_for_init)
        
        # Neuron activity tracking for dead neuron detection
        self.neuron_activity_history = deque(maxlen=12500)  # Track last 12,500 batch activations
        self.steps_since_update = 0
        
    def _init_weights(self, data_for_init=None):
        """Initialize weights with small random values."""
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='linear')
        
        # Initialize centering bias to geometric median of dataset if provided
        if data_for_init is not None:
            self.centering_bias.data = self._compute_geometric_median(data_for_init)
    
    def _compute_geometric_median(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the geometric median of the dataset for bias initialization."""
        # Simple approximation: use median of each dimension
        # For a more accurate geometric median, a more complex algorithm would be needed
        with torch.no_grad():
            return torch.median(data, dim=0)[0]
    
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
        # Apply centering bias to input (pre-encoder bias)
        centered_x = x - self.centering_bias
        
        # Encode to get sparse features
        sparse_code = F.relu(self.encoder(centered_x))
        
        # Apply top-k sparsity if enabled
        if self.use_top_k:
            sparse_code = self._apply_top_k(sparse_code)
        
        # Decode to reconstruct input
        decoded = self.decoder(sparse_code)
        
        # Add centering bias to output (post-decoder bias)
        reconstructed = decoded + self.centering_bias
        
        return reconstructed, sparse_code
    
    def track_neuron_activity(self, sparse_code: torch.Tensor):
        """
        Track which neurons are active in the current batch.
        
        Args:
            sparse_code: Tensor of shape [batch_size, hidden_dim] with sparse activations
        """
        # For each neuron, check if it fired for any example in the batch
        active_neurons = (sparse_code > 0).any(dim=0).cpu()  # Shape: [hidden_dim]
        self.neuron_activity_history.append(active_neurons)
        self.steps_since_update += 1
        
    def identify_dead_neurons(self) -> torch.Tensor:
        """
        Identify neurons that haven't fired in the tracked history.
        
        Returns:
            Boolean tensor with True for dead neurons [hidden_dim]
        """
        if not self.neuron_activity_history:
            return torch.zeros(self.hidden_dim, dtype=torch.bool)
            
        # Combine all activation records
        activity_tensor = torch.stack(list(self.neuron_activity_history), dim=0)  # [history_len, hidden_dim]
        
        # A neuron is dead if it never fired for any input in the history
        dead_neurons = ~activity_tensor.any(dim=0)  # [hidden_dim]
        
        return dead_neurons
        
    def resample_dead_neurons(
        self, 
        inputs: torch.Tensor, 
        optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, int]:
        """
        Resample dead neurons based on inputs with high reconstruction loss.
        
        Args:
            inputs: Tensor of input samples [n_samples, input_dim]
            optimizer: The optimizer being used for training (to reset parameters)
            
        Returns:
            Tuple of (loss_values, number_of_resampled_neurons)
        """
        # Identify dead neurons
        dead_neurons = self.identify_dead_neurons()
        dead_neuron_indices = torch.where(dead_neurons)[0]
        n_dead = len(dead_neuron_indices)
        
        if n_dead == 0:
            return torch.tensor(0.0, device=inputs.device), 0
            
        # Calculate loss for each input to find high-loss samples
        with torch.no_grad():
            batch_size = 128  # Process in batches to avoid OOM
            losses = []
            
            for i in range(0, inputs.size(0), batch_size):
                batch = inputs[i:i+batch_size]
                reconstructed, _ = self.forward(batch)
                batch_loss = F.mse_loss(reconstructed, batch, reduction='none').sum(dim=1)  # [batch_size]
                losses.append(batch_loss)
                
            all_losses = torch.cat(losses, dim=0)  # [n_samples]
            
            # Square the losses to increase weight of high-loss samples
            sampling_weights = all_losses ** 2
            sampling_probs = sampling_weights / sampling_weights.sum()
            
        # Sample inputs based on their loss
        indices = torch.multinomial(sampling_probs, num_samples=n_dead, replacement=True)
        sampled_inputs = inputs[indices]  # [n_dead, input_dim]
        
        # Compute average norm of alive encoder weights
        alive_norms = torch.norm(self.encoder.weight[~dead_neurons], dim=1)
        avg_alive_norm = alive_norms.mean() if alive_norms.numel() > 0 else torch.tensor(1.0, device=inputs.device)
        
        
        with torch.no_grad():
            for i, neuron_idx in enumerate(dead_neuron_indices):
                # Normalize the input and set as decoder weights
                input_vec = sampled_inputs[i] - self.centering_bias  # Apply centering bias 
                normalized_input = F.normalize(input_vec, dim=0) * avg_alive_norm * 0.2
                
                # Set encoder weights directly from the normalized input
                self.encoder.weight[neuron_idx] = normalized_input
                self.encoder.bias[neuron_idx] = 0.0
                
                # Set decoder weights (transposed relationship)
                self.decoder.weight[:, neuron_idx] = F.normalize(input_vec, dim=0)
                
                # Reset optimizer state for this neuron's parameters
                param_id_encoder_w = id(self.encoder.weight)
                param_id_encoder_b = id(self.encoder.bias)
                param_id_decoder_w = id(self.decoder.weight)
                
                if param_id_encoder_w in optimizer.state:
                    for key in optimizer.state[param_id_encoder_w]:
                        if hasattr(optimizer.state[param_id_encoder_w][key], 'index_select'):
                            optimizer.state[param_id_encoder_w][key][neuron_idx] = 0
                
                if param_id_encoder_b in optimizer.state:
                    for key in optimizer.state[param_id_encoder_b]:
                        if hasattr(optimizer.state[param_id_encoder_b][key], 'index_select'):
                            optimizer.state[param_id_encoder_b][key][neuron_idx] = 0
                            
                if param_id_decoder_w in optimizer.state:
                    for key in optimizer.state[param_id_decoder_w]:
                        if hasattr(optimizer.state[param_id_decoder_w][key], 'index_select'):
                            optimizer.state[param_id_decoder_w][key][:, neuron_idx] = 0
        
        # Reset activity tracking after resampling
        self.neuron_activity_history.clear()
        self.steps_since_update = 0
        
        return all_losses[indices].mean(), n_dead
    
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