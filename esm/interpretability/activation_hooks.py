import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np
from collections import defaultdict
import functools


class ActivationCaptureHook:
    """
    Hook to capture activations from model layers.
    
    Args:
        storage (dict): Dictionary to store captured activations
        key (str): Key to use for storing the activations
        post_process_fn (callable, optional): Function to post-process activations
    """
    def __init__(
        self, 
        storage: Dict[str, Any], 
        key: str, 
        post_process_fn: Optional[Callable] = None
    ):
        self.storage = storage
        self.key = key
        self.post_process_fn = post_process_fn
        
    def __call__(self, module, input_args, output):
        """Capture the output of the module."""
        if isinstance(output, tuple):
            # Some modules return multiple outputs, take the first one
            activation = output[0]
        else:
            activation = output
            
        # Apply post-processing if provided
        if self.post_process_fn is not None:
            activation = self.post_process_fn(activation)
            
        self.storage[self.key] = activation


class ESMCActivationCapturer:
    """
    Hooks into a ESMC model to capture activations.
    
    Args:
        model: ESMC model to hook into
        layers_to_capture: List of layer indices to capture from (default: all layers)
        component: Which component to analyze ('attention', 'mlp', 'embeddings')
    """
    def __init__(
        self, 
        model,
        layers_to_capture: Optional[List[int]] = None,
        component: str = 'mlp'
    ):
        self.model = model
        self.component = component
        
        # Default to capturing all layers if not specified
        if layers_to_capture is None:
            layers_to_capture = list(range(len(model.transformer.blocks)))
        self.layers_to_capture = layers_to_capture
        
        # Store hooks and activations
        self.hooks = []
        self.activations = {}
        
    def _hook_fn(self, layer_idx: int, module: nn.Module, input: Any, output: Any) -> None:
        """Hook function to capture activations."""
        # Make a copy to avoid modifying the original tensor
        self.activations[f"layer_{layer_idx}_{self.component}"] = output.detach()
        
    def attach_hooks(self) -> None:
        """Attach hooks to the model."""
        self.remove_hooks()  # Clear any existing hooks
        
        for layer_idx in self.layers_to_capture:
            # Get the appropriate module based on the component
            if self.component == 'mlp':
                if hasattr(self.model.transformer.blocks[layer_idx], 'mlp'):
                    module = self.model.transformer.blocks[layer_idx].mlp
                else:
                    # If direct MLP access not available, find the equivalent module
                    module = self.model.transformer.blocks[layer_idx].ffn
            elif self.component == 'attention':
                module = self.model.transformer.blocks[layer_idx].attention
            elif self.component == 'embeddings':
                # For embeddings, we use the output of the transformer's first layer
                if layer_idx == 0:
                    module = self.model.embed
                else:
                    continue  # Skip other layers for embeddings component
            else:
                raise ValueError(f"Unknown component: {self.component}")
            
            # Register the hook
            hook = module.register_forward_hook(
                functools.partial(self._hook_fn, layer_idx)
            )
            self.hooks.append(hook)
    
    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.activations = {}
        
    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get the captured activations."""
        return self.activations 