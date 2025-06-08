"""
wise/WISE.py - Main WISE implementation for GPT-2 editing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import copy
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from .utils import parent_module, EditingMeanAct, EarlyStopMeter
from .merge import get_merge_algorithm
from .config import WISEConfig


class WISEAdapter(nn.Module):
    """Adapter module for WISE editing."""
    
    def __init__(self, original_layer: nn.Module, config: WISEConfig):
        super().__init__()
        self.original_layer = original_layer
        self.config = config
        
        # Store original weights
        self.register_buffer('original_weight', original_layer.weight.data.clone())
        if original_layer.bias is not None:
            self.register_buffer('original_bias', original_layer.bias.data.clone())
        else:
            self.original_bias = None
        
        # Learnable delta weights
        self.delta_weight = nn.Parameter(torch.zeros_like(original_layer.weight))
        if original_layer.bias is not None:
            self.delta_bias = nn.Parameter(torch.zeros_like(original_layer.bias))
        else:
            self.delta_bias = None
        
        # Activation tracking
        self.activation_mean = EditingMeanAct()
        
    def forward(self, x):
        """Forward pass with delta weights."""
        # Apply original + delta weights
        weight = self.original_weight + self.delta_weight
        bias = None
        if self.original_bias is not None:
            bias = self.original_bias + self.delta_bias
        
        # Linear transformation
        output = F.linear(x, weight, bias)
        
        # Track activations during training
        if self.training:
            act_value = torch.mean(torch.abs(output)).item()
            self.activation_mean.update(act_value)
        
        return output
    
    def get_delta_norm(self):
        """Get L2 norm of delta weights."""
        delta_norm = torch.norm(self.delta_weight)
        if self.delta_bias is not None:
            delta_norm += torch.norm(self.delta_bias)
        return delta_norm
    
    def apply_mask(self, mask_ratio: float = 0.1):
        """Apply random masking to delta weights."""
        with torch.no_grad():
            mask = torch.rand_like(self.delta_weight) > mask_ratio
            self.delta_weight.data *= mask.float()
            
            if self.delta_bias is not None:
                bias_mask = torch.rand_like(self.delta_bias) > mask_ratio
                self.delta_bias.data *= bias_mask.float()
    
    def merge_weights(self, merge_ratio: float = 0.5):
        """Merge delta weights into original weights."""
        with torch.no_grad():
            # Update original weights
            self.original_weight.data += merge_ratio * self.delta_weight.data
            if self.original_bias is not None and self.delta_bias is not None:
                self.original_bias.data += merge_ratio * self.delta_bias.data
            
            # Reset delta weights
            self.delta_weight.data.zero_()
            if self.delta_bias is not None:
                self.delta_bias.data.zero_()
    
    def reset_deltas(self):
        """Reset delta weights to zero."""
        with torch.no_grad():
            self.delta_weight.data.zero_()
            if self.delta_bias is not None:
                self.delta_bias.data.zero_()


class WISE:
    """Main WISE editing controller."""
    
    def __init__(self, config: WISEConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # WISE adapters
        self.adapters = {}
        self.original_modules = {}
        
        # Training state
        self.edit_history = []
        self.merge_algorithm = get_merge_algorithm(config.merge_alg)
        
        # Install adapters
        self._install_adapters()
        
        # Move to device
        self.model.to(self.device)
    
    def _load_model(self) -> GPT2LMHeadModel:
        """Load GPT-2 model."""
        if self.config.model_path:
            model = GPT2LMHeadModel.from_pretrained(self.config.model_path)
        else:
            model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        return model
    
    def _load_tokenizer(self) -> GPT2Tokenizer:
        """Load GPT-2 tokenizer."""
        if self.config.model_path:
            tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _install_adapters(self):
        """Install WISE adapters on specified layers."""
        for param_name in self.config.inner_params:
            # Get parent module and parameter name
            parent = parent_module(self.model, param_name)
            param_parts = param_name.split('.')
            layer_name = param_parts[-2]  # e.g., 'c_fc' from 'transformer.h.0.mlp.c_fc.weight'
            
            # Get original layer
            original_layer = getattr(parent, layer_name)
            
            # Create adapter
            adapter = WISEAdapter(original_layer, self.config)
            
            # Replace layer with adapter
            setattr(parent, layer_name, adapter)
            
            # Store references
            self.adapters[param_name] = adapter
            self.original_modules[param_name] = original_layer
    
    def edit(self, 
             input_text: str, 
             target_text: str, 
             **kwargs) -> Dict[str, float]:
        """
        Perform WISE editing on the model.
        
        Args:
            input_text: Input prompt text
            target_text: Target completion text
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with loss metrics
        """
        self.model.train()
        
        # Tokenize inputs
        input_ids = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )['input_ids'].to(self.device)
        
        target_ids = self.tokenizer(
            target_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )['input_ids'].to(self.device)
        
        # Combine input and target
        full_ids = torch.cat([input_ids, target_ids], dim=1)
        labels = full_ids.clone()
        labels[:, :input_ids.size(1)] = -100  # Ignore input tokens in loss
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            [p for adapter in self.adapters.values() for p in adapter.parameters()],
            lr=self.config.edit_lr
        )
        
        # Early stopping
        early_stop = EarlyStopMeter(patience=5)
        
        losses = []
        
        for iteration in range(self.config.n_iter):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(full_ids, labels=labels)
            
            # Primary loss (language modeling)
            lm_loss = outputs.loss
            
            # Activation loss
            act_loss = self._compute_activation_loss()
            
            # Regularization loss
            reg_loss = self._compute_regularization_loss()
            
            # Total loss
            total_loss = (lm_loss + 
                         self.config.alpha * act_loss + 
                         self.config.gamma * reg_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Apply norm constraint if specified
            if self.config.norm_constraint:
                for adapter in self.adapters.values():
                    torch.nn.utils.clip_grad_norm_(
                        adapter.parameters(), 
                        self.config.norm_constraint
                    )
            
            optimizer.step()
            
            # Track losses
            current_loss = total_loss.item()
            losses.append(current_loss)
            
            # Apply masking periodically
            if iteration % 5 == 0 and self.config.mask_ratio > 0:
                for adapter in self.adapters.values():
                    adapter.apply_mask(self.config.mask_ratio)
            
            # Check early stopping
            early_stop.update(current_loss)
            if early_stop.stop():
                print(f"Early stopping at iteration {iteration}")
                break
            
            # Merge weights periodically
            if (iteration + 1) % self.config.merge_freq == 0:
                self._merge_weights()
        
        # Final merge
        self._merge_weights()
        
        # Store edit history
        edit_info = {
            'input_text': input_text,
            'target_text': target_text,
            'final_loss': losses[-1] if losses else 0.0,
            'iterations': len(losses)
        }
        self.edit_history.append(edit_info)
        
        return {
            'final_loss': losses[-1] if losses else 0.0,
            'lm_loss': lm_loss.item(),
            'act_loss': act_loss,
            'reg_loss': reg_loss,
            'iterations': len(losses)
        }
    
    def _compute_activation_loss(self) -> float:
        """Compute activation-based loss."""
        act_loss = 0.0
        
        for adapter in self.adapters.values():
            if adapter.activation_mean.mean_acts:
                mean_act = adapter.activation_mean.mean()
                min_act = adapter.activation_mean.min_act()
                
                # Encourage activations to be within reasonable range
                if mean_act > 0:
                    act_loss += self.config.beta * (mean_act - min_act) ** 2
        
        return act_loss
    
    def _compute_regularization_loss(self) -> float:
        """Compute regularization loss on delta weights."""
        reg_loss = 0.0
        
        for adapter in self.adapters.values():
            reg_loss += adapter.get_delta_norm()
        
        return reg_loss
    
    def _merge_weights(self):
        """Merge delta weights using configured algorithm."""
        for param_name, adapter in self.adapters.items():
            if hasattr(adapter, 'delta_weight') and torch.norm(adapter.delta_weight) > 1e-6:
                # Simple merge for now - can be extended with more sophisticated algorithms
                adapter.merge_weights(self.config.weights)
    
    def batch_edit(self, 
                   edit_examples: List[Tuple[str, str]],
                   **kwargs) -> List[Dict[str, float]]:
        """
        Perform batch editing.
        
        Args:
            edit_examples: List of (input_text, target_text) tuples
            **kwargs: Additional arguments
        
        Returns:
            List of loss dictionaries for each edit
        """
        results = []
        
        for i, (input_text, target_text) in enumerate(edit_examples):
            print(f"Processing edit {i+1}/{len(edit_examples)}")
            
            result = self.edit(input_text, target_text, **kwargs)
            results.append(result)
            
            # Reset activations between edits
            for adapter in self.adapters.values():
                adapter.activation_mean.reset()
        
        return results
    
    def generate(self, 
                 prompt: str, 
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_p: float = 0.9) -> str:
        """
        Generate text using the edited model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        with torch.no_grad():
            input_ids = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True
            )['input_ids'].to(self.device)
            
            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def evaluate(self, test_examples: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate the edited model.
        
        Args:
            test_examples: List of (input, expected_output) tuples
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        
        with torch.no_grad():
            for input_text, expected_output in test_examples:
                # Generate prediction
                generated = self.generate(input_text, max_length=50, temperature=0.1)
                
                # Check if prediction matches expected (simple string matching)
                if expected_output.lower().strip() in generated.lower().strip():
                    correct_predictions += 1
                
                # Compute loss
                input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids'].to(self.device)
                target_ids = self.tokenizer(expected_output, return_tensors='pt')['input_ids'].to(self.device)
                full_ids = torch.cat([input_ids, target_ids], dim=1)
                labels = full_ids.clone()
                labels[:, :input_ids.size(1)] = -100
                
                outputs = self.model(full_ids, labels=labels)
                total_loss += outputs.loss.item()
        
        accuracy = correct_predictions / len(test_examples)
        avg_loss = total_loss / len(test_examples)
        
        return {
            'accuracy': accuracy,
            'average_loss': avg_loss,
            'correct_predictions': correct_predictions,
            'total_examples': len(test_examples)
        }
    
    def save_model(self, save_path: str):
        """Save the edited model."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save edit history
        import json
        with open(f"{save_path}/edit_history.json", 'w') as f:
            json.dump(self.edit_history, f, indent=2)
    
    def reset_model(self):
        """Reset model to original state."""
        for adapter in self.adapters.values():
            adapter.reset_deltas()
            adapter.activation_mean.reset()
        
        self.edit_history.clear()
    
    def get_edit_statistics(self) -> Dict[str, Union[int, float]]:
        """Get statistics about performed edits."""
        if not self.edit_history:
            return {'total_edits': 0}
        
        losses = [edit['final_loss'] for edit in self.edit_history]
        iterations = [edit['iterations'] for edit in self.edit_history]
        
        return {
            'total_edits': len(self.edit_history),
            'average_loss': np.mean(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses),
            'average_iterations': np.mean(iterations),
            'total_iterations': np.sum(iterations)
        }
