# Utility function to build tokenizer
def build_tokenizer(tokenizer_type: str, train_path: str, min_char_freq: int = 2, tiktoken_encoding: str = "gpt2"):
    """
    Build and return a tokenizer instance.
    Args:
        tokenizer_type: 'char' or 'tiktoken'
        train_path: Path to training data (CSV)
        min_char_freq: Minimum character frequency for char tokenizer
        tiktoken_encoding: Encoding name for TikTokenTokenizer
    Returns:
        tokenizer: CharTokenizer or TikTokenTokenizer
    """
    if tokenizer_type == 'char':
        tokenizer = CharTokenizer(min_freq=min_char_freq)
        # Build vocab from training data
        import pandas as pd
        df = pd.read_csv(train_path, nrows=10000)  # Use a subset for speed
        texts = df['text'].dropna().astype(str).tolist()
        tokenizer.build_vocab(texts)
        return tokenizer
    elif tokenizer_type == 'tiktoken':
        tokenizer = TikTokenTokenizer(encoding_name=tiktoken_encoding)
        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
"""
Data loading and tokenization for TinyStories dataset
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
import mlx.core as mx
from tqdm import tqdm


class CharTokenizer:
    """Character-level tokenizer for TinyStories dataset"""
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Initialize special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in special_tokens:
            self.char_to_idx[token] = len(self.char_to_idx)
            self.idx_to_char[len(self.idx_to_char)] = token
        self.vocab_size = len(self.char_to_idx)
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts
        
        Args:
            texts: List of text strings
        """
        print("Building character vocabulary...")
        char_freq = Counter()
        
        # Count character frequencies
        for text in tqdm(texts, desc="Counting characters"):
            char_freq.update(text)
        
        # Add characters that meet minimum frequency
        for char, freq in sorted(char_freq.items()):
            if freq >= self.min_freq:
                if char not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)
        print(f"Vocabulary size: {self.vocab_size} characters")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token indices
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.char_to_idx[self.BOS_TOKEN])
        
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]))
        
        if add_special_tokens:
            tokens.append(self.char_to_idx[self.EOS_TOKEN])
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text
        
        Args:
            token_ids: List of token indices
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        special_tokens = {self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}
        chars = []
        
        for idx in token_ids:
            char = self.idx_to_char.get(idx, self.UNK_TOKEN)
            if skip_special_tokens and char in special_tokens:
                continue
            chars.append(char)
        
        return ''.join(chars)
    
    def get_pad_token_id(self) -> int:
        """Get padding token ID"""
        return self.char_to_idx[self.PAD_TOKEN]


class TikTokenTokenizer:
    """Wrapper around tiktoken for BPE tokenization"""
    
    def __init__(self, encoding_name: str = "gpt2"):
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.tokenizer.n_vocab
        except ImportError:
            raise ImportError("tiktoken not installed. Install with: pip install tiktoken")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token indices"""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token indices to text"""
        return self.tokenizer.decode(token_ids)
    
    def get_pad_token_id(self) -> int:
        """Get padding token ID (tiktoken doesn't have one, use 0)"""
        return 0


class TinyStoriesDataset:
    """Dataset for TinyStories loaded from local CSV files"""
    
    def __init__(
        self, 
        csv_path: str,
        tokenizer,
        seq_length: int = 512,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            csv_path: Path to CSV file
            tokenizer: Tokenizer instance (CharTokenizer or TikTokenTokenizer)
            seq_length: Maximum sequence length
            max_samples: Maximum number of samples to load (for testing)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.get_pad_token_id()
        
        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, nrows=max_samples)
        
        # Assume the CSV has a 'text' column
        if 'text' not in df.columns:
            # Try to find the text column
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'story' in col.lower()]
            if text_cols:
                df = df.rename(columns={text_cols[0]: 'text'})
            else:
                raise ValueError(f"Could not find text column in CSV. Columns: {df.columns.tolist()}")
        
        self.texts = df['text'].tolist()
        print(f"Loaded {len(self.texts)} samples")

        # Tokenize all texts, skip empty or invalid
        print("Tokenizing texts...")
        self.tokenized_texts = []
        skipped = 0
        for text in tqdm(self.texts, desc="Tokenizing"):
            if isinstance(text, str) and text.strip():
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                # Only keep if at least two tokens (BOS/EOS or more)
                if len(tokens) >= 2:
                    self.tokenized_texts.append(tokens)
                else:
                    skipped += 1
            else:
                skipped += 1
        print(f"Dataset ready with {len(self.tokenized_texts)} tokenized samples (skipped {skipped} empty/invalid)")
    
    def __len__(self) -> int:
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """
        Get a single sample
        
        Returns:
            input_ids: Input token sequence [seq_length]
            target_ids: Target token sequence [seq_length] (shifted by 1)
        """
        tokens = self.tokenized_texts[idx]
        
        # Truncate or pad to seq_length + 1 (we need one extra for targets)
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[:self.seq_length + 1]
        else:
            # Pad with pad token
            tokens = tokens + [self.pad_token_id] * (self.seq_length + 1 - len(tokens))
        
        # Create input and target sequences
        input_ids = mx.array(tokens[:-1], dtype=mx.int32)  # [seq_length]
        target_ids = mx.array(tokens[1:], dtype=mx.int32)  # [seq_length]
        
        return input_ids, target_ids


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 32,
    seq_length: int = 512,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[List, List]:
    """
    Create training and validation dataloaders
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_length: Sequence length
        max_train_samples: Max training samples (for testing)
        max_val_samples: Max validation samples (for testing)
        
    Returns:
        train_batches: List of training batches
        val_batches: List of validation batches
    """
    # Create datasets
    train_dataset = TinyStoriesDataset(train_path, tokenizer, seq_length, max_train_samples)
    val_dataset = TinyStoriesDataset(val_path, tokenizer, seq_length, max_val_samples)
    
    def train_batch_generator():
        print("Creating training batches (generator)...")
        indices = np.random.permutation(len(train_dataset))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_inputs = []
            batch_targets = []
            for idx in batch_indices:
                input_ids, target_ids = train_dataset[int(idx)]
                batch_inputs.append(input_ids)
                batch_targets.append(target_ids)
            batch_inputs = mx.stack(batch_inputs)
            batch_targets = mx.stack(batch_targets)
            yield (batch_inputs, batch_targets)

    def val_batch_generator():
        print("Creating validation batches (generator)...")
        for i in range(0, len(val_dataset), batch_size):
            batch_inputs = []
            batch_targets = []
            for j in range(i, min(i + batch_size, len(val_dataset))):
                input_ids, target_ids = val_dataset[j]
                batch_inputs.append(input_ids)
                batch_targets.append(target_ids)
            batch_inputs = mx.stack(batch_inputs)
            batch_targets = mx.stack(batch_targets)
            yield (batch_inputs, batch_targets)

    return train_batch_generator(), val_batch_generator()
