## ESM C Interpretability <a name="esm-c-interpretability"></a>

While ESM C provides state-of-the-art protein representations, understanding how these models represent biological features remains a challenge. We provide tools for interpreting ESMC models using Sparse Autoencoders (SAE) to extract interpretable features from model activations.

### Sparse Autoencoders for Feature Extraction <a name="sparse-autoencoders"></a>

Sparse Autoencoders learn to represent neural network activations as a sparse linear combination of interpretable features. When applied to ESMC, these features can reveal biologically meaningful patterns that the model has learned during pre-training.

Our SAE implementation:
- Extracts and processes activations from different ESMC layers and components (MLP, attention, embeddings)
- Trains autoencoders to reconstruct these activations using sparse coding
- Supports both L1 sparsity and top-k activation constraints
- Provides tools for feature visualization and interpretation

This work is hugely based on the work of **Anthropic**, especially this piece of research : https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-untied.



### Training & Usage <a name="sae-training-usage"></a>

To train SAEs on ESMC model activations:

```bash

# Train SAEs on ESMC 600M
python train_sae.py --model-name esmc_600m --data-path your_sequences.txt --output-dir ./sae_results --layers 5,15,25 --component mlp --epochs 10 --use-wandb
```

You can visualize training progress and results with Weights & Biases integration.

The resulting trained autoencoders can be used to:
- Identify neuron functions and specializations
- Discover emergent features in the protein representation space
- Analyze how representations change across different network depths
- Find neurons that activate for specific protein motifs or structures

See the `esm/interpretability` directory for the implementation details of the ESMCInterpreter class and related utilities.

**More is to come in order to intrepret trained SAE. The project now is able to train SAE, but the work on the results hasn't been done.**
