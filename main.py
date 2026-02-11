"""
Main entry point for training Gated Attention Transformer

Usage:
    python main.py --epochs 10 --batch_size 32
    python main.py --max_train_samples 1000 --epochs 5  # Quick test
"""
import argparse
import mlx.core as mx
from pathlib import Path

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.model import GatedTransformer
from src.data import build_tokenizer, create_dataloaders
from src.train import train


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Gated Attention Transformer')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--eval_batches', type=int, default=100, help='Batches for evaluation')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--save_dir', type=str, default='./results/checkpoints', help='Save directory')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, default='./data/raw/train.csv', help='Training data path')
    parser.add_argument('--val_path', type=str, default='./data/raw/validation.csv', help='Validation data path')
    parser.add_argument('--tokenizer_type', type=str, default='char', choices=['char', 'tiktoken'], 
                        help='Tokenizer type')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--min_char_freq', type=int, default=2, help='Min character frequency')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Max training samples (for testing)')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Max validation samples (for testing)')
    
    # Other arguments
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    mx.random.seed(args.seed)
    
    print("=" * 80)
    print("Gated Attention Transformer Training")
    print("=" * 80)
    
    # Build tokenizer
    print("\n1. Building tokenizer...")
    tokenizer = build_tokenizer(
        tokenizer_type=args.tokenizer_type,
        train_path=args.train_path,
        min_char_freq=args.min_char_freq
    )
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )
    
    # Create training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Initialize model
    print("\n2. Initializing model...")
    model = GatedTransformer(model_config)
    num_params = model.get_num_params()
    print(f"Model created with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # Load data
    print("\n3. Loading and preparing data...")
    train_batches, val_batches = create_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Train model
    print("\n4. Starting training...")
    train(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        config=training_config,
        resume_from=args.resume_from
    )
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Model saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
