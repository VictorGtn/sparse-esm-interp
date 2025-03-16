#!/usr/bin/env python
# coding: utf-8

"""
Training script for Sparse Autoencoders on ESMC model activations.
This script collects activations from protein sequences, trains sparse autoencoders
and interprets the learned features.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from tqdm.auto import tqdm
import os
import random
from typing import List, Dict, Optional, Tuple
import wandb  # Import wandb for experiment tracking
from esm.pretrained import load_local_model
from interpretability.esmc_interpreter import ESMCInterpreter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoders on ESM model activations")
    
    # Model and data arguments
    parser.add_argument("--model-name", type=str, default="esmc_600m",
                        help="Name of pretrained ESM model to use")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to file with protein sequences (one per line)")
    parser.add_argument("--output-dir", type=str, default="./sae_results",
                        help="Directory to save results")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Number of sequences to use from dataset (for faster experimentation)")
    parser.add_argument("--include-special-tokens", action="store_true",
                        help="Include special tokens (BOS/EOS) in collected activations")
    
    # Autoencoder arguments
    parser.add_argument("--sae-hidden-dim", type=int, default=None,
                        help="Hidden dimension for SAE (default: 2x model dim)")
    parser.add_argument("--sae-l1-coefficient", type=float, default=1e-3,
                        help="L1 sparsity coefficient")
    parser.add_argument("--use-top-k", action="store_true",
                        help="Use top-k sparsity instead of L1")
    parser.add_argument("--top-k-percentage", type=float, default=0.1, 
                        help="Percentage of features to keep active in top-k sparsity")
    parser.add_argument("--unit-norm-constraint", action="store_true", default=True,
                        help="Apply unit norm constraint to dictionary vectors with gradient projection")
    
    # Training arguments
    parser.add_argument("--layers", type=str, default="5,10,15",
                        help="Comma-separated list of layer indices to analyze")
    parser.add_argument("--component", type=str, default="mlp",
                        choices=["mlp", "attention", "embeddings"],
                        help="Component to analyze")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    
    # Dead neuron resampling arguments
    parser.add_argument("--resample-dead-neurons", action="store_true",
                        help="Enable dead neuron resampling during training")
    parser.add_argument("--resample-steps", type=str, default="25000,50000,75000,100000",
                        help="Comma-separated list of steps at which to check and resample dead neurons")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    
    # Weights & Biases arguments
    parser.add_argument("--use-wandb", action="store_true",
                        help="Track experiments with Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="esmc-sparse-autoencoders",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity name (username or team name)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Name for this specific run (default: auto-generated)")
    
    return parser.parse_args()


def load_protein_sequences(data_path: str, max_sequences: Optional[int] = None) -> List[str]:
    """
    Load protein sequences from a file.
    
    Args:
        data_path: Path to file with sequences (one per line)
        max_sequences: Maximum number of sequences to load
        
    Returns:
        List of protein sequences
    """
    logger.info(f"Loading protein sequences from {data_path}")
    
    sequences = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_sequences is not None and i >= max_sequences:
                break
            
            # Remove trailing whitespace and skip empty lines
            seq = line.strip()
            if seq:
                sequences.append(seq)
    
    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


def setup_interpreter(args):
    """
    Set up the ESM model and interpreter.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple of (model, interpreter)
    """
    logger.info(f"Loading pretrained model: {args.model_name}")
    
    # Load model
    model = load_local_model(args.model_name, args.device)
    model = model.to(args.device)
    model.eval()  # Set to evaluation mode
    
    
    # Create interpreter with specified parameters
    interpreter = ESMCInterpreter(
        model=model,
        hidden_dim=args.sae_hidden_dim,
        device=torch.device(args.device),
        l1_coefficient=args.sae_l1_coefficient if not args.use_top_k else 0.0,
        use_top_k=args.use_top_k,
        top_k_percentage=args.top_k_percentage,
    )
    
    return model, interpreter


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize Weights & Biases
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
        logger.info(f"Tracking experiment with Weights & Biases: {wandb.run.name}")
    
    # Load model and create interpreter
    model, interpreter = setup_interpreter(args)
    
    # Load protein sequences
    sequences = load_protein_sequences(args.data_path, args.subset_size)
    
    # Parse layer indices
    layer_indices = [int(idx) for idx in args.layers.split(",")]
    
    # Collect activations
    logger.info(f"Collecting activations for layers {layer_indices} ({args.component})")
    activations = interpreter.collect_activations(
        input_sequences=sequences,
        layer_indices=layer_indices,
        component=args.component,
        batch_size=args.batch_size,
        include_special_tokens=args.include_special_tokens
    )
    
    logger.info(f"Total activations collected: {len(activations)}")
    # Train autoencoders for each layer
    for layer_idx in layer_indices:
        layer_key = f"layer_{layer_idx}_{args.component}"
        if layer_key not in activations:
            logger.warning(f"No activations found for {layer_key}, skipping")
            continue
        
        logger.info(f"Training autoencoder for {layer_key}")
        layer_acts = activations[layer_key]
        
        # Create save path
        sparsity_type = "top_k" if args.use_top_k else "l1"
        save_dir = output_dir / f"layer_{layer_idx}_{args.component}_{sparsity_type}"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "autoencoder.pt"
        
        # Train autoencoder
        logger.info(f"Training autoencoder for {layer_key}")
        
        # Parse resample steps from string to list of integers
        resample_steps = None
        if args.resample_dead_neurons:
            resample_steps = [int(step) for step in args.resample_steps.split(',')]
            logger.info(f"Will resample dead neurons at steps: {resample_steps}")
        
        # Log unit norm constraint approach
        if args.unit_norm_constraint:
            logger.info("Using improved unit norm constraint with gradient projection")
        
        metrics = interpreter.train_layer_autoencoder(
            layer_idx=layer_idx,
            activations=layer_acts,
            component=args.component,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            save_path=save_path,
            use_wandb=args.use_wandb,
            wandb_prefix=f"layer_{layer_idx}/{args.component}",
            resample_dead_neurons=args.resample_dead_neurons,
            resample_steps=resample_steps,
            unit_norm_constraint=args.unit_norm_constraint,
        )
        
        # Save training metrics
        metrics_path = save_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            # Convert tensor values to float for JSON serialization
            serializable_metrics = {}
            for k, v in metrics.items():
                serializable_metrics[k] = [float(x) if not isinstance(x, float) else x for x in v]
            json.dump(serializable_metrics, f, indent=2)
        
        # Plot training metrics
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(metrics["total"])
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        
        plt.subplot(2, 2, 2)
        plt.plot(metrics["reconstruction"])
        plt.title("Reconstruction Loss")
        plt.xlabel("Epoch")
        
        if not args.use_top_k:
            plt.subplot(2, 2, 3)
            plt.plot(metrics["l1_sparsity"])
            plt.title("L1 Sparsity Loss")
            plt.xlabel("Epoch")
        
        plt.subplot(2, 2, 4)
        plt.plot(metrics["sparsity"])
        plt.title("Sparsity (% zeros)")
        plt.xlabel("Epoch")
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_metrics.png")
        
        # Log figure to wandb if enabled
        if args.use_wandb:
            wandb.log({f"layer_{layer_idx}/{args.component}/training_plot": wandb.Image(plt)})
        
        plt.close()
        
        # Interpret features
        logger.info(f"Interpreting features for {layer_key}")
        results = interpreter.interpret_features(
            layer_idx=layer_idx,
            component=args.component,
            # Use a small number of sequences for finding activating examples
            sample_inputs=sequences[:100] if len(sequences) > 100 else sequences,
            visualize=True
        )
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(results["feature_importance"])), 
                sorted(results["feature_importance"], reverse=True))
        plt.title(f"Feature Importance Distribution - {layer_key}")
        plt.xlabel("Feature Rank")
        plt.ylabel("Importance")
        plt.savefig(save_dir / "feature_importance.png")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({f"layer_{layer_idx}/{args.component}/feature_importance": wandb.Image(plt)})
        
        plt.close()
        
        # Save some sample decoder weights
        top_features = sorted(
            list(results["interpreted_features"].keys()),
            key=lambda idx: results["interpreted_features"][idx]["importance"],
            reverse=True
        )[:5]
        
        plt.figure(figsize=(15, 10))
        for i, feat_idx in enumerate(top_features):
            plt.subplot(len(top_features), 1, i+1)
            weights = results["interpreted_features"][feat_idx]["decoder_weights"]
            plt.plot(weights)
            plt.title(f"Feature {feat_idx} - Importance: {results['interpreted_features'][feat_idx]['importance']:.4f}")
        
        plt.tight_layout()
        plt.savefig(save_dir / "top_features_weights.png")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({f"layer_{layer_idx}/{args.component}/top_features": wandb.Image(plt)})
        
        plt.close()
    
    logger.info(f"Training and analysis complete. Results saved to {output_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 