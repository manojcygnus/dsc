"""
experiments/evaluate.py - Evaluation script for WISE models
"""
import torch
import argparse
import json
import os
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import WISE components
import sys
sys.path.append('..')

from wise.WISE import WISE
from wise.config import WISEConfig, load_config
from wise.utils import setup_logging
from data.dataset import load_edit_data, create_factual_edit_data


class WISEEvaluator:
    """Evaluator for WISE models."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to saved WISE model
            config_path: Path to configuration file
        """
        self.model_path = model_path
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            config_dict = load_config(config_path)
            self.config = WISEConfig.from_dict(config_dict.get('wise', {}))
        else:
            self.config = WISEConfig()
        
        # Load model
        self.wise_model = self._load_wise_model()
        
        # Load baseline model for comparison
        self.baseline_model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        self.baseline_model.to(self.config.device)
        self.baseline_model.eval()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_wise_model(self) -> WISE:
        """Load WISE model from path."""
        # Initialize WISE with config
        wise_model = WISE(self.config)
        
        # Load the saved model state
        model_state_path = os.path.join(self.model_path, "pytorch_model.bin")