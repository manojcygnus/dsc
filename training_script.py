"""
experiments/train_wise.py - Main training script for WISE editing
"""
import torch
import argparse
import json
import os
from pathlib import Path
import logging
from typing import List, Tuple

# Import WISE components
import sys
sys.path.append('..')

from wise.WISE import WISE
from wise.config import WISEConfig, TrainingConfig, load_config
from wise.utils import setup_logging, count_parameters, format_number
from data.dataset import (
    load_edit_data, 
    create_sample_edit_data, 
    create_factual_edit_data,
    split_data
)


def setup_experiment_dir(output_dir: str) -> str:
    """Setup experiment directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def load_training_data(data_config: dict) -> List[Tuple[str, str]]:
    """Load and prepare training data."""
    data_path = data_config.get('dataset_path')
    
    if data_path and os.path.exists(data_path):
        # Load from file
        data = load_edit_data(data_path)
        examples = [(item['input'], item['target']) for item in data]
    else:
        # Use sample data
        print("Using sample factual editing data...")
        sample_data = create_factual_edit_data()
        examples = [(item['input'], item['target']) for item in sample_data]
    
    return examples


def train_wise_model(wise_config: WISEConfig, 
                    training_config: TrainingConfig,
                    train_examples: List[Tuple[str, str]],
                    val_examples: List[Tuple[str, str]],
                    output_dir: str,
                    logger: logging.Logger) -> WISE:
    """Train WISE model."""
    
    # Initialize WISE
    logger.info("Initializing WISE model...")
    wise_model = WISE(wise_config)
    
    # Print model info
    param_info = count_parameters(wise_model.model)
    logger.info(f"Model parameters: {format_number(param_info['total'])} total, "
               f"{format_number(param_info['trainable'])} trainable")
    
    # Training loop
    logger.info(f"Starting training with {len(train_examples)} examples...")
    
    best_val_loss = float('inf')
    training_results = []
    
    for epoch in range(training_config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{training_config.num_epochs}")
        
        epoch_results = []
        
        # Train on examples
        for i, (input_text, target_text) in enumerate(train_examples):
            logger.info(f"Training example {i+1}/{len(train_examples)}")
            logger.info(f"Input: {input_text}")
            logger.info(f"Target: {target_text}")
            
            # Perform edit
            result = wise_model.edit(input_text, target_text)
            epoch_results.append(result)
            
            logger.info(f"Final loss: {result['final_loss']:.6f}")
            logger.info(f"Iterations: {result['iterations']}")
            
            # Save checkpoint periodically
            if (i + 1) % training_config.save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_step{i+1}")
                wise_model.save_model(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Validation
        if val_examples:
            logger.info("Running validation...")
            val_metrics = wise_model.evaluate(val_examples)
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation loss: {val_metrics['average_loss']:.6f}")
            
            # Save best model
            if val_metrics['average_loss'] < best_val_loss:
                best_val_loss = val_metrics['average_loss']
                best_model_path = os.path.join(output_dir, "best_model")
                wise_model.save_model(best_model_path)
                logger.info(f"New best model saved to {best_model_path}")
        
        # Log epoch statistics
        avg_loss = sum(r['final_loss'] for r in epoch_results) / len(epoch_results)
        avg_iterations = sum(r['iterations'] for r in epoch_results) / len(epoch_results)
        
        epoch_stats = {
            'epoch': epoch + 1,
            'average_loss': avg_loss,
            'average_iterations': avg_iterations,
            'validation_accuracy': val_metrics['accuracy'] if val_examples else None,
            'validation_loss': val_metrics['average_loss'] if val_examples else None
        }
        
        training_results.append(epoch_stats)
        logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.6f}, "
                   f"Avg Iterations: {avg_iterations:.1f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    wise_model.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training results
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Print final statistics
    edit_stats = wise_model.get_edit_statistics()
    logger.info("=== Final Training Statistics ===")
    for key, value in edit_stats.items():
        logger.info(f"{key}: {value}")
    
    return wise_model


def test_model_generation(wise_model: WISE, 
                         test_prompts: List[str],
                         logger: logging.Logger):
    """Test model generation capabilities."""
    logger.info("=== Testing Model Generation ===")
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nTest {i+1}:")
        logger.info(f"Prompt: {prompt}")
        
        generated = wise_model.generate(
            prompt, 
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        logger.info(f"Generated: {generated}")


def main():
    parser = argparse.ArgumentParser(description="Train WISE model for GPT-2 editing")
    parser.add_argument("--config", type=str, default="configs/wise_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/wise_experiment",
                       help="Output directory for results")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (optional)")
    parser.add_argument("--test_generation", action="store_true",
                       help="Test model generation after training")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup experiment directory
    output_dir = setup_experiment_dir(args.output_dir)
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting WISE training experiment")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    if os.path.exists(args.config):
        config_dict = load_config(args.config)
        wise_config = WISEConfig.from_dict(config_dict.get('wise', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
    else:
        logger.warning(f"Config file {args.config} not found. Using default configuration.")
        wise_config = WISEConfig()
        training_config = TrainingConfig()
    
    # Override data path if provided
    if args.data_path:
        training_config.dataset_path = args.data_path
    
    # Log configuration
    logger.info("=== Configuration ===")
    logger.info(f"Model: {wise_config.model_name}")
    logger.info(f"Device: {wise_config.device}")
    logger.info(f"Edit LR: {wise_config.edit_lr}")
    logger.info(f"Iterations: {wise_config.n_iter}")
    logger.info(f"Merge algorithm: {wise_config.merge_alg}")
    logger.info(f"Inner params: {wise_config.inner_params}")
    
    # Save configuration
    config_save_path = os.path.join(output_dir, "config.json")
    config_to_save = {
        'wise': wise_config.to_dict(),
        'training': training_config.__dict__
    }
    with open(config_save_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Load training data
    logger.info("Loading training data...")
    all_examples = load_training_data(training_config.__dict__)
    logger.info(f"Loaded {len(all_examples)} examples")
    
    # Split data
    train_examples, val_examples, test_examples = split_data(
        all_examples,
        train_ratio=training_config.train_split,
        val_ratio=training_config.val_split,
        test_ratio=training_config.test_split
    )
    
    logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")
    
    # Train model
    wise_model = train_wise_model(
        wise_config,
        training_config,
        train_examples,
        val_examples,
        output_dir,
        logger
    )
    
    # Final evaluation on test set
    if test_examples:
        logger.info("=== Final Test Evaluation ===")
        test_metrics = wise_model.evaluate(test_examples)
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test loss: {test_metrics['average_loss']:.6f}")
        
        # Save test results
        test_results_path = os.path.join(output_dir, "test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    # Test generation capabilities
    if args.test_generation:
        test_prompts = [
            "The capital of France is",
            "The largest planet in our solar system is",
            "Water boils at",
            "The author of Romeo and Juliet is",
            "The chemical symbol for gold is"
        ]
        test_model_generation(wise_model, test_prompts, logger)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
