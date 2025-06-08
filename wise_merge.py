"""
wise/merge.py - Weight merging algorithms for WISE
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union


class BaseMerge:
    """Base class for weight merging algorithms."""
    
    def execute(self, weights: Union[float, List[float]], base_weight: torch.Tensor, 
                task_weights: List[torch.Tensor], densities: Optional[List[float]] = None):
        """
        Execute the merging algorithm.
        
        Args:
            weights: Merging weights/ratios
            base_weight: Base model weights
            task_weights: List of task-specific weights
            densities: Optional density parameters
        
        Returns:
            Merged weight tensor
        """
        raise NotImplementedError


class slerp(BaseMerge):
    """Spherical Linear Interpolation merging."""
    
    def execute(self, weight: float, base_weight: torch.Tensor, 
                task_weights: List[torch.Tensor], densities: Optional[List[float]] = None):
        """
        Perform SLERP between base and task weights.
        
        Args:
            weight: Interpolation weight (0 to 1)
            base_weight: Base model weights
            task_weights: List containing single task weight tensor
            densities: Not used in SLERP
        
        Returns:
            Interpolated weight tensor
        """
        if len(task_weights) != 1:
            raise ValueError("SLERP only supports merging with one task weight")
        
        task_weight = task_weights[0]
        
        # Flatten tensors for computation
        base_flat = base_weight.flatten()
        task_flat = task_weight.flatten()
        
        # Compute dot product
        dot = torch.dot(base_flat, task_flat)
        
        # Compute norms
        base_norm = torch.norm(base_flat)
        task_norm = torch.norm(task_flat)
        
        # Normalize
        base_unit = base_flat / base_norm
        task_unit = task_flat / task_norm
        
        # Compute angle
        cos_angle = torch.clamp(torch.dot(base_unit, task_unit), -1.0, 1.0)
        angle = torch.acos(cos_angle)
        
        # SLERP interpolation
        if torch.sin(angle) < 1e-6:  # Vectors are nearly parallel
            result = (1 - weight) * base_flat + weight * task_flat
        else:
            result = (torch.sin((1 - weight) * angle) / torch.sin(angle)) * base_flat + \
                    (torch.sin(weight * angle) / torch.sin(angle)) * task_flat
        
        return result.reshape(base_weight.shape)


class linear(BaseMerge):
    """Linear interpolation merging."""
    
    def execute(self, weight: Union[float, List[float]], base_weight: torch.Tensor, 
                task_weights: List[torch.Tensor], densities: Optional[List[float]] = None):
        """
        Perform linear interpolation between weights.
        
        Args:
            weight: Interpolation weight(s)
            base_weight: Base model weights
            task_weights: List of task weight tensors
            densities: Not used in linear interpolation
        
        Returns:
            Interpolated weight tensor
        """
        if isinstance(weight, (int, float)):
            if len(task_weights) != 1:
                raise ValueError("Single weight value requires exactly one task weight")
            return (1 - weight) * base_weight + weight * task_weights[0]
        
        # Multiple weights case
        if len(weight) != len(task_weights):
            raise ValueError("Number of weights must match number of task weights")
        
        result = base_weight.clone()
        total_weight = sum(weight)
        
        if total_weight > 0:
            base_contribution = 1 - total_weight
            result = base_contribution * base_weight
            
            for w, task_w in zip(weight, task_weights):
                result += w * task_w
        
        return result


class GTA(BaseMerge):
    """
    General Task Arithmetic (GTA) merging with various strategies.
    Supports TIES, DARE, and magnitude-based merging.
    """
    
    def __init__(self, selection_method: Optional[str] = None, 
                 aggregation_method: Optional[str] = None, 
                 normalize: bool = False):
        """
        Initialize GTA merger.
        
        Args:
            selection_method: 'magnitude', 'random', 'rescaled_random', or None
            aggregation_method: 'sum', 'mean', or None
            normalize: Whether to normalize the result
        """
        self.selection_method = selection_method
        self.aggregation_method = aggregation_method
        self.normalize = normalize
    
    def execute(self, weights: List[float], base_weight: torch.Tensor, 
                task_weights: List[torch.Tensor], densities: Optional[List[float]] = None):
        """
        Execute GTA merging.
        
        Args:
            weights: List of merging weights
            base_weight: Base model weights
            task_weights: List of task weight tensors
            densities: Density parameters for sparsification
        
        Returns:
            Merged weight tensor
        """
        if len(weights) != len(task_weights):
            raise ValueError("Number of weights must match number of task weights")
        
        # Compute task vectors (differences from base)
        task_vectors = [task_w - base_weight for task_w in task_weights]
        
        # Apply selection method
        if self.selection_method == 'magnitude':
            task_vectors = self._magnitude_selection(task_vectors, densities)
        elif self.selection_method == 'random':
            task_vectors = self._random_selection(task_vectors, densities)
        elif self.selection_method == 'rescaled_random':
            task_vectors = self._rescaled_random_selection(task_vectors, densities)
        
        # Apply weights
        weighted_vectors = [w * tv for w, tv in zip(weights, task_vectors)]
        
        # Aggregate
        if self.aggregation_method == 'sum':
            merged_vector = sum(weighted_vectors)
        elif self.aggregation_method == 'mean':
            merged_vector = sum(weighted_vectors) / len(weighted_vectors)
        else:
            merged_vector = sum(weighted_vectors)
        
        # Apply to base
        result = base_weight + merged_vector
        
        # Normalize if requested
        if self.normalize:
            result = self._normalize_weights(result, base_weight)
        
        return result
    
    def _magnitude_selection(self, task_vectors: List[torch.Tensor], 
                           densities: Optional[List[float]]):
        """Select parameters based on magnitude."""
        if densities is None:
            return task_vectors
        
        selected_vectors = []
        for tv, density in zip(task_vectors, densities):
            if density >= 1.0:
                selected_vectors.append(tv)
                continue
            
            # Compute magnitude and select top parameters
            magnitude = torch.abs(tv)
            flat_mag = magnitude.flatten()
            k = int(density * len(flat_mag))
            
            if k == 0:
                selected_vectors.append(torch.zeros_like(tv))
                continue
            
            # Get top-k indices
            _, indices = torch.topk(flat_mag, k)
            
            # Create mask
            mask = torch.zeros_like(flat_mag, dtype=torch.bool)
            mask[indices] = True
            mask = mask.reshape(tv.shape)
            
            # Apply mask
            selected_tv = tv.clone()
            selected_tv[~mask] = 0
            selected_vectors.append(selected_tv)
        
        return selected_vectors
    
    def _random_selection(self, task_vectors: List[torch.Tensor], 
                         densities: Optional[List[float]]):
        """Randomly select parameters (DARE method)."""
        if densities is None:
            return task_vectors
        
        selected_vectors = []
        for tv, density in zip(task_vectors, densities):
            if density >= 1.0:
                selected_vectors.append(tv)
                continue
            
            # Random mask
            mask = torch.rand_like(tv) < density
            selected_tv = tv.clone()
            selected_tv[~mask] = 0
            selected_vectors.append(selected_tv)
        
        return selected_vectors
    
    def _rescaled_random_selection(self, task_vectors: List[torch.Tensor], 
                                  densities: Optional[List[float]]):
        """Randomly select and rescale parameters."""
        if densities is None:
            return task_vectors
        
        selected_vectors = []
        for tv, density in zip(task_vectors, densities):
            if density >= 1.0:
                selected_vectors.append(tv)
                continue
            
            # Random mask with rescaling
            mask = torch.rand_like(tv) < density
            selected_tv = tv.clone()
            selected_tv[~mask] = 0
            
            # Rescale by density to maintain expected magnitude
            if density > 0:
                selected_tv = selected_tv / density
            
            selected_vectors.append(selected_tv)
        
        return selected_vectors
    
    def _normalize_weights(self, merged_weight: torch.Tensor, 
                          base_weight: torch.Tensor):
        """Normalize merged weights."""
        base_norm = torch.norm(base_weight)
        merged_norm = torch.norm(merged_weight)
        
        if merged_norm > 0:
            return merged_weight * (base_norm / merged_norm)
        else:
            return merged_weight


# Create merge algorithm instances
merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}


def get_merge_algorithm(name: str) -> BaseMerge:
    """
    Get merge algorithm by name.
    
    Args:
        name: Algorithm name
    
    Returns:
        Merge algorithm instance
    """
    if name not in merge_dict:
        raise ValueError(f"Unknown merge algorithm: {name}. "
                        f"Available: {list(merge_dict.keys())}")
    
    return merge_dict[name]


def list_merge_algorithms():
    """List all available merge algorithms."""
    return list(merge_dict.keys())
