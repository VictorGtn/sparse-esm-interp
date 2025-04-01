from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import esm as esm
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from interpretability.activation_hooks import ESMCActivationCapturer
from interpretability.sparse_autoencoder import SparseAutoencoder


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
        model,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        l1_coefficient: float = 1e-3,
        use_top_k: bool = False,
        top_k_percentage: float = 0.1,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device

        self.model_dim = model.embed.embedding_dim
        self.hidden_dim = hidden_dim or (2 * self.model_dim)
        self.l1_coefficient = l1_coefficient
        self.use_top_k = use_top_k
        self.top_k_percentage = top_k_percentage

        self.capturer = ESMCActivationCapturer(model)

        self.autoencoders = {}

        self.dtype = next(model.parameters()).dtype

    def create_autoencoder_for_layer(
        self,
        layer_idx: int,
        component: str = "mlp",
        data_for_init: Optional[torch.Tensor] = None,
    ) -> SparseAutoencoder:
        """
        Create a sparse autoencoder for a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            component: Which component to analyze ('attention', 'mlp', 'embeddings')
            data_for_init: Optional sample of activations to initialize the bias term

        Returns:
            SparseAutoencoder instance
        """
        key = f"{component}_{layer_idx}"

        if component == "embeddings":
            input_dim = self.model_dim
        elif component == "attention":
            input_dim = self.model_dim
        elif component == "mlp":
            input_dim = self.model_dim
        else:
            raise ValueError(f"Unknown component: {component}")

        autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            l1_coefficient=self.l1_coefficient,
            use_top_k=self.use_top_k,
            top_k_percentage=self.top_k_percentage,
            dtype=self.dtype,
            data_for_init=data_for_init,
        ).to(self.device)

        self.autoencoders[key] = autoencoder
        return autoencoder

    def collect_activations(
        self,
        input_sequences: List[str],
        layer_indices: List[int],
        component: str = "mlp",
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
        self.capturer = ESMCActivationCapturer(
            self.model, layers_to_capture=layer_indices, component=component
        )
        self.capturer.attach_hooks()

        all_activations = defaultdict(list)

        for i in tqdm(
            range(0, len(input_sequences), batch_size), desc="Collecting activations"
        ):
            batch = input_sequences[i : i + batch_size]

            self.capturer.clear_activations()

            batch_tokens = [self.model.tokenizer.encode(seq) for seq in batch]
            max_len = max(len(tokens) for tokens in batch_tokens)

            padded_tokens = [
                tokens + [self.model.tokenizer.pad_token_id] * (max_len - len(tokens))
                for tokens in batch_tokens
            ]

            tokens_tensor = torch.tensor(padded_tokens, device=self.device)

            with torch.no_grad():
                self.model(tokens_tensor)

                activations = self.capturer.get_activations()
                for k, v in activations.items():
                    for j, tokens in enumerate(batch_tokens):
                        seq_len = len(tokens)

                        start_idx = 0 if include_special_tokens else 1
                        end_idx = seq_len if include_special_tokens else seq_len - 1
                        seq_acts = v[j, start_idx:end_idx]
                        all_activations[k].append(seq_acts)

        self.capturer.remove_hooks()

        processed_activations = {}
        for k, act_list in all_activations.items():
            concatenated = torch.cat(
                [act.reshape(-1, act.size(-1)) for act in act_list], dim=0
            )
            processed_activations[k] = concatenated

        return processed_activations

    def train_layer_autoencoder(
        self,
        layer_idx: int,
        train_activations: torch.Tensor,
        valid_activations: torch.Tensor,
        component: str = "mlp",
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        save_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_prefix: str = "",
        resample_dead_neurons: bool = True,
        resample_steps: List[int] = [25000, 50000, 75000, 100000],
        unit_norm_constraint: bool = True,
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
            resample_dead_neurons: Whether to resample dead neurons during training
            resample_steps: Steps at which to resample dead neurons
            unit_norm_constraint: Whether to apply unit norm constraints on dictionary vectors

        Returns:
            Dictionary of training metrics
        """
        key = f"{component}_{layer_idx}"

        if key not in self.autoencoders:
            sample_size = min(1000, train_activations.shape[0])
            activation_sample = train_activations[:sample_size]
            self.create_autoencoder_for_layer(
                layer_idx, component, data_for_init=activation_sample
            )

        autoencoder = self.autoencoders[key]
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)

        train_dataset = TensorDataset(train_activations)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        valid_dataset = TensorDataset(valid_activations)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )

        if use_wandb:
            try:
                import wandb
            except ImportError:
                print(
                    "wandb not installed. Run 'pip install wandb' to enable wandb logging."
                )
                use_wandb = False

        if wandb_prefix and not wandb_prefix.endswith("/"):
            wandb_prefix = wandb_prefix + "/"

        if resample_dead_neurons:
            resampling_size = min(819200, len(train_activations))
            resampling_indices = torch.randperm(len(train_activations))[
                :resampling_size
            ]
            resampling_activations = train_activations[resampling_indices].to(
                self.device
            )

        metrics = defaultdict(list)
        global_step = 0

        if unit_norm_constraint:
            autoencoder.normalize_decoder_weights()

        for epoch in range(epochs):
            epoch_metrics = defaultdict(float)
            n_batches = 0

            for (x,) in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x = x.to(self.device)
                reconstructed, sparse_code = autoencoder(x)

                autoencoder.track_neuron_activity(sparse_code)

                loss_dict = autoencoder.loss(x, reconstructed, sparse_code)
                loss = loss_dict["total"]

                optimizer.zero_grad()
                loss.backward()

                if unit_norm_constraint:
                    autoencoder.project_decoder_grad()
                optimizer.step()

                for k, v in loss_dict.items():
                    epoch_metrics[k] += v.item() if torch.is_tensor(v) else v
                n_batches += 1

                global_step += 1

                if (
                    resample_dead_neurons
                    and global_step in resample_steps
                    and autoencoder.steps_since_update >= 12500
                ):
                    resample_loss, n_resampled = autoencoder.resample_dead_neurons(
                        resampling_activations, optimizer
                    )

                    if n_resampled > 0:
                        if use_wandb:
                            wandb.log(
                                {
                                    f"{wandb_prefix}resampled_neurons": n_resampled,
                                    f"{wandb_prefix}resampling_loss": resample_loss.item(),
                                    f"{wandb_prefix}step": global_step,
                                }
                            )

            for k, v in epoch_metrics.items():
                avg_v = v / n_batches
                metrics[k].append(avg_v)

            if use_wandb:
                wandb_metrics = {f"{wandb_prefix}{k}": v for k, v in metrics.items()}
                wandb_metrics[f"{wandb_prefix}epoch"] = epoch
                wandb.log(wandb_metrics)

            print(
                f"Epoch {epoch + 1}: recon_loss={metrics['reconstruction'][-1]:.4f}, "
                f"sparsity={metrics['sparsity'][-1]:.4f}"
            )
            for (x,) in valid_dataloader:
                x = x.to(self.device)
                reconstructed, sparse_code = autoencoder(x)
                loss_dict = autoencoder.loss(x, reconstructed, sparse_code)
                loss = loss_dict["total"]
                for k, v in loss_dict.items():
                    metrics[k].append(v.item())
                if use_wandb:
                    wandb.log(
                        {
                            f"{wandb_prefix}valid_loss": loss.item(),
                        }
                    )

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(autoencoder.state_dict(), path)

        return dict(metrics)
