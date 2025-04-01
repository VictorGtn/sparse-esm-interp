#!/usr/bin/env python
# coding: utf-8

"""
Training script for Sparse Autoencoders on ESMC model activations.
This script collects activations from protein sequences, trains sparse autoencoders
and interprets the learned features.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from esm.pretrained import load_local_model

from interpretability.esmc_interpreter import ESMCInterpreter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration dictionary
args = {
    # Model settings
    "model_name": "esmc_600m",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    # Data settings
    "training_data_path": "data/train_data.csv",  # Path to training data
    "valid_data_path": "data/valid_data.csv",  # Path to validation data
    "subset_size": None,
    "include_special_tokens": False,
    # Output settings
    "output_dir": "./sae_results",
    # SAE settings
    "sae_hidden_dim": None,
    "sae_l1_coefficient": 1e-3,
    "use_top_k": False,
    "top_k_percentage": 0.1,
    "unit_norm_constraint": True,
    # Training settings
    "layers": "5,10,15",
    "component": "mlp",
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "resample_dead_neurons": False,
    "resample_steps": "25000,50000,75000,100000",
    # Weights & Biases settings
    "use_wandb": False,
    "wandb_project": "esmc-sparse-autoencoders",
    "wandb_entity": None,
    "wandb_name": None,
}


def load_protein_sequences(
    data_path: str, max_sequences: Optional[int] = None
) -> List[str]:
    """
    Load protein sequences from a CSV file.

    Args:
        data_path: Path to CSV file with sequences (must have 'text' column)
        max_sequences: Maximum number of sequences to load

    Returns:
        List of protein sequences
    """
    logger.info(f"Loading protein sequences from {data_path}")

    # Read CSV file using pandas
    df = pd.read_csv(data_path)

    # Get sequences from 'text' column
    sequences = df["text"].tolist()

    # Limit sequences if max_sequences is specified
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


def setup_interpreter(args):
    """
    Set up the ESM model and interpreter.

    Args:
        args: Configuration dictionary

    Returns:
        tuple of (model, interpreter)
    """
    logger.info(f"Loading pretrained model: {args['model_name']}")

    model = load_local_model(args["model_name"], args["device"])
    model = model.to(args["device"])
    model.eval()

    interpreter = ESMCInterpreter(
        model=model,
        hidden_dim=args["sae_hidden_dim"],
        device=torch.device(args["device"]),
        l1_coefficient=args["sae_l1_coefficient"] if not args["use_top_k"] else 0.0,
        use_top_k=args["use_top_k"],
        top_k_percentage=args["top_k_percentage"],
    )

    return model, interpreter


def main():
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w") as f:
        json.dump(args, f, indent=2)

    if args["use_wandb"]:
        wandb.init(
            project=args["wandb_project"],
            entity=args["wandb_entity"],
            name=args["wandb_name"],
            config=args,
        )
        logger.info(f"Tracking experiment with Weights & Biases: {wandb.run.name}")

    model, interpreter = setup_interpreter(args)
    train_sequences = load_protein_sequences(
        args["training_data_path"], args["subset_size"]
    )
    valid_sequences = load_protein_sequences(
        args["valid_data_path"], args["subset_size"]
    )

    layer_indices = [int(idx) for idx in args["layers"].split(",")]

    logger.info(
        f"Collecting activations for layers {layer_indices} ({args['component']})"
    )
    train_activations = interpreter.collect_activations(
        input_sequences=train_sequences,
        layer_indices=layer_indices,
        component=args["component"],
        batch_size=args["batch_size"],
        include_special_tokens=args["include_special_tokens"],
    )

    valid_activations = interpreter.collect_activations(
        input_sequences=valid_sequences,
        layer_indices=layer_indices,
        component=args["component"],
        batch_size=args["batch_size"],
        include_special_tokens=args["include_special_tokens"],
    )

    logger.info(
        f"Total activations collected: {len(train_activations)} training, {len(valid_activations)} validation"
    )
    for layer_idx in layer_indices:
        layer_key = f"layer_{layer_idx}_{args['component']}"
        if layer_key not in train_activations:
            logger.warning(f"No activations found for {layer_key}, skipping")
            continue

        logger.info(f"Training autoencoder for {layer_key}")
        train_layer_acts = train_activations[layer_key]
        valid_layer_acts = valid_activations[layer_key]
        sparsity_type = "top_k" if args["use_top_k"] else "l1"
        save_dir = output_dir / f"layer_{layer_idx}_{args['component']}_{sparsity_type}"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "autoencoder.pt"

        logger.info(f"Training autoencoder for {layer_key}")

        resample_steps = None
        if args["resample_dead_neurons"]:
            resample_steps = [int(step) for step in args["resample_steps"].split(",")]
            logger.info(f"Will resample dead neurons at steps: {resample_steps}")

        if args["unit_norm_constraint"]:
            logger.info("Using improved unit norm constraint with gradient projection")

        metrics = interpreter.train_layer_autoencoder(
            layer_idx=layer_idx,
            train_activations=train_layer_acts,
            valid_activations=valid_layer_acts,
            component=args["component"],
            epochs=args["epochs"],
            batch_size=args["batch_size"],
            lr=args["learning_rate"],
            save_path=save_path,
            use_wandb=args["use_wandb"],
            wandb_prefix=f"layer_{layer_idx}/{args['component']}",
            resample_dead_neurons=args["resample_dead_neurons"],
            resample_steps=resample_steps,
            unit_norm_constraint=args["unit_norm_constraint"],
        )

        metrics_path = save_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            serializable_metrics = {}
            for k, v in metrics.items():
                serializable_metrics[k] = [
                    float(x) if not isinstance(x, float) else x for x in v
                ]
            json.dump(serializable_metrics, f, indent=2)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(metrics["total"])
        plt.title("Total Loss")
        plt.xlabel("Epoch")

        plt.subplot(2, 2, 2)
        plt.plot(metrics["reconstruction"])
        plt.title("Reconstruction Loss")
        plt.xlabel("Epoch")

        if not args["use_top_k"]:
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

        if args["use_wandb"]:
            wandb.log(
                {
                    f"layer_{layer_idx}/{args['component']}/training_plot": wandb.Image(
                        plt
                    )
                }
            )

        plt.close()

    logger.info(f"Training and analysis complete. Results saved to {output_dir}")

    if args["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
