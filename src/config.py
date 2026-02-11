"""
Configuration for Gated Attention Transformer
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""
    vocab_size: int = 10000  # Will be set based on tokenizer
    d_model: int = 512  # Embedding dimension
    n_layers: int = 6  # Number of transformer blocks
    n_heads: int = 8  # Number of attention heads
    d_ff: int = 2048  # Feedforward dimension (4x d_model)
    max_seq_length: int = 512  # Maximum sequence length
    dropout: float = 0.1  # Dropout rate
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 500  # Steps between evaluations
    eval_batches: int = 100  # Number of batches for evaluation
    
    # Checkpointing
    save_interval: int = 1000  # Steps between checkpoints
    save_dir: str = "./results/checkpoints"
    
    # Data limits (for testing)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    train_path: str = "./data/raw/train.csv"
    val_path: str = "./data/raw/validation.csv"
    tokenizer_type: str = "char"  # "char" or "tiktoken"
    seq_length: int = 512
    num_workers: int = 0  # MLX doesn't use multiprocessing like PyTorch
    
    # Character tokenizer settings
    min_char_freq: int = 2  # Minimum frequency for character to be in vocab
    
    # TikToken settings
    tiktoken_encoding: str = "gpt2"  # GPT-2 BPE encoding
