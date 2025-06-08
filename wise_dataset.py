"""
data/dataset.py - Dataset classes for WISE editing
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import json
import pandas as pd
from typing import List, Dict, Optional, Union
import random
from pathlib import Path


class WISEEditDataset(Dataset):
    """Dataset for WISE editing examples."""
    
    def __init__(self, 
                 data: List[Dict[str, str]], 
                 tokenizer: GPT2Tokenizer,
                 max_length: int = 512,
                 prompt_template: Optional[str] = None):
        """
        Initialize WISE editing dataset.
        
        Args:
            data: List of editing examples with 'input' and 'target' keys
            tokenizer: GPT2 tokenizer
            max_length: Maximum sequence length
            prompt_template: Optional template for formatting prompts
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "{input}"
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Format input
        input_text = self.prompt_template.format(**example)
        target_text = example['target']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'input_text': input_text,
            'target_text': target_text
        }


class WISETextDataset(Dataset):
    """Dataset for general text data for WISE training."""
    
    def __init__(self,
                 texts: List[str],
                 tokenizer: GPT2Tokenizer,
                 max_length: int = 512,
                 stride: int = 256):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: GPT2 tokenizer
            max_length: Maximum sequence length
            stride: Stride for sliding window (for long texts)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Process texts into chunks
        self.text_chunks = self._process_texts(texts)
    
    def _process_texts(self, texts: List[str]) -> List[str]:
        """Process texts into manageable chunks."""
        chunks = []
        
        for text in texts:
            # Tokenize to check length
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) <= self.max_length:
                chunks.append(text)
            else:
                # Split into overlapping chunks
                for i in range(0, len(tokens), self.stride):
                    chunk_tokens = tokens[i:i + self.max_length]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)
        
        return chunks
    
    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        text = self.text_chunks[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': text
        }


class WISEConversationDataset(Dataset):
    """Dataset for conversation-style editing."""
    
    def __init__(self,
                 conversations: List[Dict[str, Any]],
                 tokenizer: GPT2Tokenizer,
                 max_length: int = 512,
                 system_prompt: str = ""):
        """
        Initialize conversation dataset.
        
        Args:
            conversations: List of conversation dictionaries
            tokenizer: GPT2 tokenizer
            max_length: Maximum sequence length
            system_prompt: System prompt to prepend
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format conversation into a single string."""
        formatted = self.system_prompt
        
        if 'messages' in conversation:
            for message in conversation['messages']:
                role = message.get('role', 'user')
                content = message.get('content', '')
                formatted += f"\n{role.capitalize()}: {content}"
        elif 'input' in conversation and 'output' in conversation:
            formatted += f"\nUser: {conversation['input']}\nAssistant: {conversation['output']}"
        
        return formatted.strip()
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        formatted_text = self._format_conversation(conversation)
        
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': formatted_text,
            'conversation': conversation
        }


def load_edit_data(data_path: str, 
                   format_type: str = 'json',
                   sample_size: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load editing data from file.
    
    Args:
        data_path: Path to data file
        format_type: Format of data ('json', 'csv', 'jsonl')
        sample_size: Optional sample size limit
    
    Returns:
        List of editing examples
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if format_type == 'json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    elif format_type == 'jsonl':
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    
    elif format_type == 'csv':
        df = pd.read_csv(data_path)
        data = df.to_dict('records')
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Sample if requested
    if sample_size and len(data) > sample_size:
        data = random.sample(data, sample_size)
    
    return data


def load_text_data(data_path: str) -> List[str]:
    """
    Load text data from file.
    
    Args:
        data_path: Path to text file
    
    Returns:
        List of text strings
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.suffix == '.txt':
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by paragraphs or sentences
        texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        
    elif data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if isinstance(data[0], str):
                texts = data
            else:
                texts = [item.get('text', '') for item in data]
        else:
            texts = [data.get('text', '')]
    
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return [text for text in texts if text.strip()]


def create_dataloader(dataset: Dataset,
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """
    Create DataLoader for WISE dataset.
    
    Args:
        dataset: WISE dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def split_data(data: List[Any], 
               train_ratio: float = 0.8,
               val_ratio: float = 0.1,
               test_ratio: float = 0.1,
               random_seed: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        data: Data to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(random_seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    n = len(data_copy)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:train_size + val_size]
    test_data = data_copy[train_size + val_size:]
    
    return train_data, val_data, test_data


# Example data creation functions
def create_sample_edit_data() -> List[Dict[str, str]]:
    """Create sample editing data for testing."""
    return [
        {
            'input': 'The capital of France is',
            'target': 'Paris'
        },
        {
            'input': 'The largest planet in our solar system is',
            'target': 'Jupiter'
        },
        {
            'input': 'Water boils at',
            'target': '100 degrees Celsius'
        },
        {
            'input': 'The author of Romeo and Juliet is',
            'target': 'William Shakespeare'
        },
        {
            'input': 'The chemical symbol for gold is',
            'target': 'Au'
        }
    ]


def create_factual_edit_data() -> List[Dict[str, str]]:
    """Create factual editing examples."""
    return [
        {
            'input': 'Question: What is the capital of Japan?\nAnswer:',
            'target': 'Tokyo'
        },
        {
            'input': 'Question: Who invented the telephone?\nAnswer:',
            'target': 'Alexander Graham Bell'
        },
        {
            'input': 'Question: What is the speed of light?\nAnswer:',
            'target': '299,792,458 meters per second'
        },
        {
            'input': 'Question: When was the first iPhone released?\nAnswer:',
            'target': '2007'
        },
        {
            'input': 'Question: What is the largest ocean on Earth?\nAnswer:',
            'target': 'Pacific Ocean'
        }
    ]


def create_conversation_data() -> List[Dict[str, Any]]:
    """Create sample conversation data."""
    return [
        {
            'messages': [
                {'role': 'user', 'content': 'Hello, how are you?'},
                {'role': 'assistant', 'content': 'I am doing well, thank you for asking!'}
            ]
        },
        {
            'messages': [
                {'role': 'user', 'content': 'Can you help me with Python?'},
                {'role': 'assistant', 'content': 'Of course! I would be happy to help you with Python programming.'}
            ]
        },
        {
            'input': 'What is machine learning?',
            'output': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.'