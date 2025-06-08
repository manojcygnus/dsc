"""
wise/utils.py - Core utility functions for WISE editing
"""
import torch
import numpy as np
from typing import Optional, Union


def parent_module(model, name):
    """
    Get the parent module of a given parameter name.
    
    Args:
        model: The PyTorch model
        name: Parameter name (e.g., 'transformer.h.0.mlp.c_fc')
    
    Returns:
        Parent module containing the target parameter
    """
    names = name.split('.')
    parent = model
    for name_part in names[:-1]:
        if name_part.isdigit():
            parent = parent[int(name_part)]
        else:
            parent = getattr(parent, name_part)
    return parent


def brackets_to_periods(name):
    """
    Convert bracket notation to period notation.
    
    Args:
        name: Parameter name with brackets (e.g., 'transformer.h[0].mlp.c_fc')
    
    Returns:
        Parameter name with periods (e.g., 'transformer.h.0.mlp.c_fc')
    """
    return name.replace('[', '.').replace(']', '')


class EarlyStopMeter:
    """
    Monitor training loss for early stopping.
    """
    def __init__(self, tolerance=1e-4, patience=5):
        self.tolerance = tolerance
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.losses = []
    
    def update(self, loss):
        """Update with new loss value."""
        self.losses.append(loss)
        
        if loss < self.best_loss - self.tolerance:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
    
    def stop(self):
        """Check if training should stop."""
        return self.counter >= self.patience


class EditingMeanAct:
    """
    Track mean activation statistics during editing.
    """
    def __init__(self, min_a=1e9):
        self.mean_acts = []
        self.min_a = min_a
    
    def update(self, act_value):
        """Update with new activation value."""
        self.mean_acts.append(act_value)
        self.min_a = min(self.min_a, act_value)
    
    def min_act(self):
        """Get minimum activation value."""
        return self.min_a if self.min_a != 1e9 else 0.0
    
    def mean(self):
        """Get mean of all activation values."""
        return np.mean(self.mean_acts) if self.mean_acts else 0.0
    
    def reset(self):
        """Reset all tracking."""
        self.mean_acts = []
        self.min_a = 1e9


def get_model_layers(model, layer_type='linear'):
    """
    Get all layers of a specific type from the model.
    
    Args:
        model: PyTorch model
        layer_type: Type of layer to find ('linear', 'conv', etc.)
    
    Returns:
        Dictionary of layer names and modules
    """
    layers = {}
    
    for name, module in model.named_modules():
        if layer_type.lower() == 'linear' and isinstance(module, torch.nn.Linear):
            layers[name] = module
        elif layer_type.lower() == 'conv1d' and hasattr(torch.nn, 'Conv1D'):
            if isinstance(module, torch.nn.Conv1D):
                layers[name] = module
    
    return layers


def print_model_structure(model, max_depth=3):
    """
    Print model structure for debugging.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to print
    """
    def _print_recursive(module, name, depth=0):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        print(f"{indent}{name}: {type(module).__name__}")
        
        if hasattr(module, 'weight') and module.weight is not None:
            print(f"{indent}  - weight shape: {module.weight.shape}")
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"{indent}  - bias shape: {module.bias.shape}")
        
        for child_name, child_module in module.named_children():
            _print_recursive(child_module, child_name, depth + 1)
    
    _print_recursive(model, "model")


def save_checkpoint(model, optimizer, config, path, additional_info=None):
    """
    Save training checkpoint.
    
    Args:
        model: WISE model
        optimizer: Optimizer state
        config: Configuration object
        path: Save path
        additional_info: Additional information to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config,
        'additional_info': additional_info or {}
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model=None, optimizer=None):
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        Loaded checkpoint data
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def setup_logging(log_level='INFO'):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wise_training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('WISE')


def count_parameters(model):
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def format_number(num):
    """Format large numbers for display."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)
