"""
wise/config.py - Configuration classes for WISE editing
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import yaml
import json


@dataclass
class WISEConfig:
    """Configuration for WISE editing."""
    
    # Model configuration
    model_name: str = "gpt2"
    model_path: Optional[str] = None
    device: str = "cuda"
    
    # Layer configuration
    inner_params: List[str] = field(default_factory=lambda: ["transformer.h.0.mlp.c_fc.weight"])
    
    # Editing hyperparameters
    edit_lr: float = 1e-4
    n_iter: int = 20
    batch_size: int = 1
    
    # Activation loss parameters
    alpha: float = 1e-6
    beta: float = 0.75
    gamma: float = 0.1
    act_ratio: float = 0.8
    
    # Weight management
    save_freq: Optional[int] = 10
    merge_freq: int = 50
    merge_alg: str = "linear"
    weights: float = 0.5
    densities: Optional[List[float]] = None
    
    # Masking
    mask_ratio: float = 0.1
    
    # Retrieval and replay
    retrieve: bool = False
    replay: bool = False
    
    # Constraints
    norm_constraint: Optional[float] = None
    
    # Training configuration
    hidden_act: str = "gelu"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_freq is not None and self.merge_freq % self.save_freq != 0:
            raise ValueError("merge_freq must be divisible by save_freq")
        
        if self.merge_alg not in ["linear", "slerp", "ties", "dare_ties", "dare_linear", 
                                  "magnitude", "magnitude_norm", "sign"]:
            raise ValueError(f"Unknown merge algorithm: {self.merge_alg}")
        
        if self.edit_lr <= 0:
            raise ValueError("edit_lr must be positive")
        
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WISEConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'WISEConfig':
        """Create config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'WISEConfig':
        """Create config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'inner_params': self.inner_params,
            'edit_lr': self.edit_lr,
            'n_iter': self.n_iter,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'act_ratio': self.act_ratio,
            'save_freq': self.save_freq,
            'merge_freq': self.merge_freq,
            'merge_alg': self.merge_alg,
            'weights': self.weights,
            'densities': self.densities,
            'mask_ratio': self.mask_ratio,
            'retrieve': self.retrieve,
            'replay': self.replay,
            'norm_constraint': self.norm_constraint,
            'hidden_act': self.hidden_act
        }
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Data configuration
    dataset_path: str = "data/training_data.json"
    max_length: int = 512
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training parameters
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Checkpointing
    output_dir: str = "outputs"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Metrics
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError("Data splits must sum to 1.0")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Input data
    data_path: str = "data/raw_data.txt"
    data_format: str = "text"  # "text", "json", "csv"
    
    # Text processing
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    
    # Data augmentation
    augment_data: bool = False
    augmentation_factor: float = 1.0
    
    # Preprocessing
    lowercase: bool = False
    remove_special_chars: bool = False
    min_text_length: int = 10
    max_text_length: int = 1000
    
    # Caching
    cache_dir: Optional[str] = "cache"
    overwrite_cache: bool = False


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        'wise': WISEConfig().to_dict(),
        'training': TrainingConfig().__dict__,
        'data': DataConfig().__dict__
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")


# Example configuration templates
EXAMPLE_CONFIGS = {
    'basic': {
        'wise': {
            'model_name': 'gpt2',
            'inner_params': ['transformer.h.0.mlp.c_fc.weight'],
            'edit_lr': 1e-4,
            'n_iter': 20,
            'merge_alg': 'linear',
            'save_freq': 10,
            'merge_freq': 50
        },
        'training': {
            'dataset_path': 'data/training_data