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
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False, dtype=dtype)
        
        self.centering_bias = nn.Parameter(torch.zeros(input_dim, dtype=dtype))
        
        self._init_weights(data_for_init)
        
        self.neuron_activity_history = deque(maxlen=12500)  # Track last 12,500 batch activations
        self.steps_since_update = 0
        
    def _init_weights(self, data_for_init=None):
        """Initialize weights with small random values."""
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='linear')
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        
        if data_for_init is not None:
            self.centering_bias.data = self._compute_geometric_median(data_for_init)
    
    def _compute_geometric_median(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the geometric median of the dataset for bias initialization."""
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
        
        values, _ = torch.topk(x, k, dim=1)
        thresholds = values[:, -1].unsqueeze(1) 
        
        return x * (x >= thresholds)
    
    def project_decoder_grad(self):
        """
        Project out gradient components parallel to dictionary vectors to maintain unit norm.
        
        This improves training by ensuring the gradient step doesn't modify the norm of
        dictionary vectors, instead of naively renormalizing after each step.
        """
        if self.decoder.weight.grad is None:
            return
            
        with torch.no_grad():
            W = self.decoder.weight.data  
            
            dW = self.decoder.weight.grad.data  

            for j in range(W.shape[1]):
                w_j = W[:, j].view(-1, 1) 
                
                dw_j = dW[:, j].view(-1, 1)  
                
                parallel_component = torch.matmul(w_j.T, dw_j) * w_j
                dW[:, j] = dw_j.view(-1) - parallel_component.view(-1)
    
    def normalize_decoder_weights(self):
        """
        Normalize the decoder weights to have unit norm.
        
        This serves as a fallback to ensure unit norm is maintained,
        but the primary constraint is handled by projecting out
        parallel gradient components.
        """
        with torch.no_grad():
            normalized_weights = F.normalize(self.decoder.weight.data, dim=0)
            self.decoder.weight.data.copy_(normalized_weights)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_input, sparse_code)
        """
        centered_x = x - self.centering_bias
        
        sparse_code = F.relu(self.encoder(centered_x))
        
        if self.use_top_k:
            sparse_code = self._apply_top_k(sparse_code)
        
        decoded = self.decoder(sparse_code)
        
        reconstructed = decoded + self.centering_bias
        
        return reconstructed, sparse_code
    
    def track_neuron_activity(self, sparse_code: torch.Tensor):
        """
        Track which neurons are active in the current batch.
        
        Args:
            sparse_code: Tensor of shape [batch_size, hidden_dim] with sparse activations
        """
        active_neurons = (sparse_code > 1e-8).any(dim=0).cpu()
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
            
        activity_tensor = torch.stack(list(self.neuron_activity_history), dim=0)  # [history_len, hidden_dim]
        
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
        dead_neurons = self.identify_dead_neurons()
        dead_neuron_indices = torch.where(dead_neurons)[0]
        n_dead = len(dead_neuron_indices)
        
        if n_dead == 0:
            return torch.tensor(0.0, device=inputs.device), 0
            
        with torch.no_grad():
            batch_size = 128  
            losses = []
            
            for i in range(0, inputs.size(0), batch_size):
                batch = inputs[i:i+batch_size]
                reconstructed, _ = self.forward(batch)
                batch_loss = F.mse_loss(reconstructed, batch, reduction='none').sum(dim=1)  # [batch_size]
                losses.append(batch_loss)
                
            all_losses = torch.cat(losses, dim=0)  # [n_samples]
            
            sampling_weights = all_losses ** 2
            sampling_probs = sampling_weights / sampling_weights.sum()
            
        indices = torch.multinomial(sampling_probs, num_samples=n_dead, replacement=True)
        sampled_inputs = inputs[indices]  
        
        alive_norms = torch.norm(self.encoder.weight[~dead_neurons], dim=1)
        avg_alive_norm = alive_norms.mean() if alive_norms.numel() > 0 else torch.tensor(1.0, device=inputs.device)
        
        with torch.no_grad():
            for i, neuron_idx in enumerate(dead_neuron_indices):
                input_vec = sampled_inputs[i] - self.centering_bias  # Apply centering bias 
                normalized_input = F.normalize(input_vec, dim=0) * avg_alive_norm * 0.2
                
                self.encoder.weight[neuron_idx] = normalized_input
                self.encoder.bias[neuron_idx] = 0.0
                
                self.decoder.weight[:, neuron_idx] = F.normalize(input_vec, dim=0)
                
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
        recon_loss = F.mse_loss(reconstructed, x)
        
        l1_loss = torch.tensor(0.0, device=x.device)
        if self.l1_coefficient > 0:
            l1_loss = self.l1_coefficient * sparse_code.abs().mean()
        
        total_loss = recon_loss + l1_loss
        
        sparsity = (sparse_code == 0).float().mean().item()
        mean_activation = sparse_code.abs().mean().item()
        
        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "l1_sparsity": l1_loss,
            "mean_activation": mean_activation,
            "sparsity": sparsity,
        }
        
