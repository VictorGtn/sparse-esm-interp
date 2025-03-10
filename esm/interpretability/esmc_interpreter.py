import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from tqdm.auto import tqdm
from collections import defaultdict

from esm.models.esmc import ESMC
from esm.interpretability.sparse_autoencoder import SparseAutoencoder
from esm.interpretability.activation_hooks import ESMCActivationCapturer


class ESMCInterpreter:
    """
    Tools for interpreting ESMC using sparse autoencoders.
    
    Args:
        model: ESMC model to interpret
        hidden_dim: Dimension for sparse features (default: twice the model dimension)
        device: Device to use
        l1_coefficient: L1 sparsity coefficient
        use_top_k: Whether to use top-k sparsity in the encoder
        top_k_percentage: Percentage of features to keep active (between 0 and 1)
    """
    def __init__(
        self, 
        model: ESMC,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        l1_coefficient: float = 1e-3,
        use_top_k: bool = False,
        top_k_percentage: float = 0.1,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Get model dimension from the embedding layer since d_model isn't directly accessible
        # The embedding dimension is the output size of the embedding layer
        self.model_dim = model.embed.embedding_dim
        self.hidden_dim = hidden_dim or (2 * self.model_dim)
        self.l1_coefficient = l1_coefficient
        self.use_top_k = use_top_k
        self.top_k_percentage = top_k_percentage
        
        # Create activation capturer
        self.capturer = ESMCActivationCapturer(model)
        
        # Dictionary to store autoencoders for different layers
        self.autoencoders = {}
        
        # Determine data type from model
        self.dtype = next(model.parameters()).dtype
        
    def create_autoencoder_for_layer(self, layer_idx: int, component: str = 'mlp') -> SparseAutoencoder:
        """
        Create a sparse autoencoder for a specific layer.
        
        Args:
            layer_idx: Index of the transformer layer
            component: Which component to analyze ('attention', 'mlp', 'embeddings')
            
        Returns:
            SparseAutoencoder instance
        """
        key = f"{component}_{layer_idx}"
        
        # Get input dimension based on component
        if component == 'embeddings':
            input_dim = self.model_dim
        elif component == 'attention':
            input_dim = self.model_dim
        elif component == 'mlp':
            input_dim = self.model_dim
        else:
            raise ValueError(f"Unknown component: {component}")
        
        # Create autoencoder
        autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            l1_coefficient=self.l1_coefficient,
            use_top_k=self.use_top_k,
            top_k_percentage=self.top_k_percentage,
            dtype=self.dtype
        ).to(self.device)
        
        self.autoencoders[key] = autoencoder
        return autoencoder
        
    def collect_activations(
        self,
        input_sequences: List[str],
        layer_indices: List[int],
        component: str = 'mlp',
        batch_size: int = 4,
        include_special_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activations from the model for a dataset of inputs.
        
        Args:
            input_sequences: List of protein sequences
            layer_indices: Which layers to collect activations from
            component: Which component to analyze ('attention', 'mlp', 'embeddings')
            batch_size: Batch size for processing
            include_special_tokens: Whether to include special tokens (BOS/EOS) in collected activations
            
        Returns:
            Dictionary mapping layer keys to tensors of activations
        """
        # Set up activation capturer
        self.capturer = ESMCActivationCapturer(
            self.model, 
            layers_to_capture=layer_indices,
            component=component
        )
        self.capturer.attach_hooks()
        
        all_activations = defaultdict(list)
        
        # Process inputs in batches
        for i in tqdm(range(0, len(input_sequences), batch_size), desc="Collecting activations"):
            batch = input_sequences[i:i+batch_size]
            
            # Clear previous activations
            self.capturer.clear_activations()
            
            # Tokenize sequences
            batch_tokens = [self.model.tokenizer.encode(seq) for seq in batch]
            max_len = max(len(tokens) for tokens in batch_tokens)
            
            # Pad sequences to the same length
            padded_tokens = [
                tokens + [self.model.tokenizer.pad_token_id] * (max_len - len(tokens))
                for tokens in batch_tokens
            ]
            
            # Create attention mask
            attention_mask = torch.tensor([
                [1] * len(tokens) + [0] * (max_len - len(tokens))
                for tokens in batch_tokens
            ], device=self.device)
            
            # Convert to tensor
            tokens_tensor = torch.tensor(padded_tokens, device=self.device)
            
            # Forward pass with no grad
            with torch.no_grad():
                self.model(tokens_tensor)
                
                # Collect activations
                activations = self.capturer.get_activations()
                for k, v in activations.items():
                    # Extract activations, excluding padding tokens
                    for j, tokens in enumerate(batch_tokens):
                        seq_len = len(tokens)
                        
                        # Determine start and end indices based on whether to include special tokens
                        start_idx = 0 if include_special_tokens else 1
                        end_idx = seq_len if include_special_tokens else seq_len - 1
                        
                        # Get activations for the specified token range
                        # For ESMC, tokens include BOS and EOS special tokens
                        seq_acts = v[j, start_idx:end_idx]
                        all_activations[k].append(seq_acts)
        
        # Remove hooks
        self.capturer.remove_hooks()
        
        # Process and concatenate all collected activations
        processed_activations = {}
        for k, act_list in all_activations.items():
            # Concatenate all activations along the first dimension
            concatenated = torch.cat([act.reshape(-1, act.size(-1)) for act in act_list], dim=0)
            processed_activations[k] = concatenated
            
        return processed_activations
    
    def train_layer_autoencoder(
        self,
        layer_idx: int,
        activations: torch.Tensor,
        component: str = 'mlp',
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        save_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_prefix: str = ""
    ) -> Dict[str, List[float]]:
        """
        Train an autoencoder for a specific layer.
        
        Args:
            layer_idx: Layer index
            activations: Tensor of activations [n_samples, hidden_dim]
            component: Component type ('attention', 'mlp', 'embeddings')
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate
            save_path: Path to save the trained autoencoder
            use_wandb: Whether to log metrics to Weights & Biases
            wandb_prefix: Prefix for wandb metric names
            
        Returns:
            Dictionary of training metrics
        """
        key = f"{component}_{layer_idx}"
        
        # Create autoencoder if it doesn't exist
        if key not in self.autoencoders:
            self.create_autoencoder_for_layer(layer_idx, component)
        
        autoencoder = self.autoencoders[key]
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)
        
        # Training loop
        metrics = defaultdict(list)
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Import wandb only if needed (to avoid dependency requirement when not using it)
        if use_wandb:
            try:
                import wandb
            except ImportError:
                print("wandb not installed. Run 'pip install wandb' to enable wandb logging.")
                use_wandb = False
        
        # Add prefix with trailing slash if provided
        if wandb_prefix and not wandb_prefix.endswith('/'):
            wandb_prefix = wandb_prefix + '/'
        
        for epoch in range(epochs):
            epoch_metrics = defaultdict(float)
            n_batches = 0
            
            for (x,) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                
                # Forward pass
                reconstructed, sparse_code = autoencoder(x)
                
                # Compute loss
                loss_dict = autoencoder.loss(x, reconstructed, sparse_code)
                loss = loss_dict["total"]
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                for k, v in loss_dict.items():
                    epoch_metrics[k] += v.item() if torch.is_tensor(v) else v
                n_batches += 1
            
            # Compute epoch averages
            for k, v in epoch_metrics.items():
                avg_v = v / n_batches
                metrics[k].append(avg_v)
            
            # Log to wandb if enabled
            if use_wandb:
                wandb_metrics = {f"{wandb_prefix}{k}": v for k, v in metrics.items()}
                # Add epoch information to metrics
                wandb_metrics[f"{wandb_prefix}epoch"] = epoch
                wandb.log(wandb_metrics)
            
            print(f"Epoch {epoch+1}: recon_loss={metrics['reconstruction'][-1]:.4f}, "
                  f"sparsity={metrics['sparsity'][-1]:.4f}")
        
        # Save model if requested
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(autoencoder.state_dict(), path)
            
        return dict(metrics)
    
