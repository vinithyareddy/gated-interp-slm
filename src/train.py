"""
Training loop and utilities for Gated Attention Transformer

Optimized for Apple Silicon using MLX framework.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple, Optional
import time
import json
import os
from pathlib import Path
from tqdm import tqdm
import math


def loss_fn(model, input_ids: mx.array, target_ids: mx.array) -> mx.array:
    """
    Compute cross-entropy loss for language modeling
    
    Args:
        model: GatedTransformer model
        input_ids: Input token indices [batch, seq_len]
        target_ids: Target token indices [batch, seq_len]
        
    Returns:
        loss: Scalar loss value
    """
    # Forward pass
    logits, _ = model(input_ids, return_interpretability=False)  # [batch, seq_len, vocab_size]
    
    # Reshape for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)  # [batch * seq_len, vocab_size]
    targets = target_ids.reshape(-1)  # [batch * seq_len]
    
    # Cross-entropy loss
    loss = nn.losses.cross_entropy(logits, targets, reduction='mean')
    
    return loss


def train_step(
    model,
    optimizer,
    input_ids: mx.array,
    target_ids: mx.array,
    grad_clip: float = 1.0
) -> Tuple[mx.array, Dict]:
    """
    Single training step
    
    Args:
        model: GatedTransformer model
        optimizer: MLX optimizer
        input_ids: Input tokens [batch, seq_len]
        target_ids: Target tokens [batch, seq_len]
        grad_clip: Gradient clipping threshold
        
    Returns:
        loss: Training loss
        metrics: Dictionary of training metrics
    """
    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, input_ids, target_ids)
    
    # Clip gradients
    if grad_clip > 0:
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=grad_clip)
    else:
        grad_norm = 0.0
    
    # Update parameters
    optimizer.update(model, grads)
    
    # Evaluate updated model and optimizer state
    mx.eval(model.parameters(), optimizer.state)
    
    # Compute perplexity
    perplexity = mx.exp(loss)
    
    metrics = {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item()
    }
    
    return loss, metrics


def evaluate(
    model,
    val_batches: List[Tuple[mx.array, mx.array]],
    max_batches: Optional[int] = None
) -> Dict:
    """
    Evaluate model on validation set
    
    Args:
        model: GatedTransformer model
        val_batches: List of validation batches
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        metrics: Dictionary with validation metrics
    """
    total_loss = 0.0
    num_batches = 0
    
    # Limit number of batches if specified
    batches_to_eval = val_batches[:max_batches] if max_batches else val_batches
    
    for input_ids, target_ids in batches_to_eval:
        loss = loss_fn(model, input_ids, target_ids)
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return {
        'val_loss': avg_loss,
        'val_perplexity': perplexity
    }


def get_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float = 1e-5
) -> float:
    """
    Cosine learning rate schedule with warmup
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        lr: Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay


def save_checkpoint(
    model,
    optimizer,
    step: int,
    epoch: int,
    metrics: Dict,
    save_path: str
):
    """
    Save model checkpoint
    
    Args:
        model: GatedTransformer model
        optimizer: MLX optimizer
        step: Current training step
        epoch: Current epoch
        metrics: Training metrics
        save_path: Path to save checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Flatten nested parameter structure
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(d, list):
            for i, v in enumerate(d):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # It's an array
            return {parent_key: d}
        return dict(items)
    
    # Get flattened parameters
    flat_params = flatten_dict(model.parameters())
    
    # Use .npz format
    save_path_npz = save_path.replace('.safetensors', '.npz')
    
    # Save model parameters
    mx.savez(save_path_npz, **flat_params)
    
    # Save metadata separately
    metadata_path = save_path_npz.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'step': step,
            'epoch': epoch,
            'metrics': metrics
        }, f, indent=2)
    
    print(f"Checkpoint saved to {save_path_npz}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str
) -> Tuple[int, int, Dict]:
    """
    Load model checkpoint
    
    Args:
        model: GatedTransformer model
        optimizer: MLX optimizer
        checkpoint_path: Path to checkpoint
        
    Returns:
        step: Training step
        epoch: Epoch number
        metrics: Training metrics
    """
    # Handle both .npz and .safetensors extensions
    if checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.safetensors', '.npz')
    
    # Load flattened parameters
    flat_params = mx.load(checkpoint_path)
    
    # Unflatten to nested structure
    def unflatten_dict(flat_dict, sep='.'):
        result = {}
        for key, value in flat_dict.items():
            parts = key.split(sep)
            d = result
            for part in parts[:-1]:
                # Check if it's a list index
                if part.isdigit():
                    # Convert to list if needed
                    if not isinstance(d, list):
                        d = []
                    idx = int(part)
                    # Extend list if needed
                    while len(d) <= idx:
                        d.append({})
                    d = d[idx]
                else:
                    if part not in d:
                        # Check if next part is a digit to determine if we need a list
                        next_idx = parts.index(part) + 1
                        if next_idx < len(parts) and parts[next_idx].isdigit():
                            d[part] = []
                        else:
                            d[part] = {}
                    d = d[part]
            
            # Set the final value
            final_key = parts[-1]
            if final_key.isdigit():
                idx = int(final_key)
                if not isinstance(d, list):
                    d = []
                while len(d) <= idx:
                    d.append(None)
                d[idx] = value
            else:
                d[final_key] = value
        
        return result
    
    # Unflatten and update model
    nested_params = unflatten_dict(flat_params)
    model.update(nested_params)
    
    # Load metadata
    metadata_path = checkpoint_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    return metadata['step'], metadata['epoch'], metadata['metrics']


def train(
    model,
    train_batches: List[Tuple[mx.array, mx.array]],
    val_batches: List[Tuple[mx.array, mx.array]],
    config,
    resume_from: Optional[str] = None
):
    """
    Main training loop
    
    Args:
        model: GatedTransformer model
        train_batches: List of training batches
        val_batches: List of validation batches
        config: TrainingConfig instance
        resume_from: Path to checkpoint to resume from
    """
    # Initialize optimizer
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step, start_epoch, _ = load_checkpoint(model, optimizer, resume_from)
        print(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    # Calculate total steps (for generator, estimate from dataset length and batch size)
    if hasattr(train_batches, '__len__'):
        steps_per_epoch = len(train_batches)
    else:
        # train_batches is a generator, so estimate steps from dataset
        if hasattr(model, 'config') and hasattr(model.config, 'train_path'):
            import pandas as pd
            df = pd.read_csv(model.config.train_path)
            num_samples = len(df)
        else:
            # Fallback: try to get from config
            num_samples = getattr(config, 'max_train_samples', None) or 0
        steps_per_epoch = math.ceil(num_samples / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    
    # Training state
    global_step = start_step
    best_val_loss = float('inf')
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
    print(f"Model parameters: {model.get_num_params():,}\n")
    
    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        epoch_start_time = time.time()

        # Training
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_batches, desc=f"Training", total=steps_per_epoch)
        batch_idx = 0
        for input_ids, target_ids in train_batches:
            # Update learning rate
            lr = get_lr_schedule(
                global_step,
                config.warmup_steps,
                total_steps,
                config.learning_rate
            )
            optimizer.learning_rate = lr

            # Training step
            loss, metrics = train_step(
                model,
                optimizer,
                input_ids,
                target_ids,
                config.grad_clip
            )

            epoch_loss += metrics['loss']
            global_step += 1
            batch_idx += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'lr': f"{lr:.2e}"
            })
            progress_bar.update(1)
            
            # Evaluation
            if global_step % config.eval_interval == 0:
                model.eval()
                val_metrics = evaluate(model, val_batches, config.eval_batches)
                model.train()
                
                print(f"\nStep {global_step}: "
                      f"val_loss={val_metrics['val_loss']:.4f}, "
                      f"val_ppl={val_metrics['val_perplexity']:.2f}")
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_path = os.path.join(config.save_dir, 'best_model.safetensors')
                    save_checkpoint(model, optimizer, global_step, epoch, val_metrics, save_path)
            
            # Periodic checkpoint
            if global_step % config.save_interval == 0:
                save_path = os.path.join(config.save_dir, f'checkpoint_step_{global_step}.safetensors')
                save_checkpoint(model, optimizer, global_step, epoch, metrics, save_path)
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / max(batch_idx, 1)

        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average training loss: {avg_epoch_loss:.4f}")
        
        # End of epoch evaluation
        model.eval()
        val_metrics = evaluate(model, val_batches)
        model.train()
        
        print(f"Validation: loss={val_metrics['val_loss']:.4f}, "
              f"perplexity={val_metrics['val_perplexity']:.2f}\n")
        
        # Save epoch checkpoint
        save_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch + 1}.safetensors')
        save_checkpoint(model, optimizer, global_step, epoch + 1, val_metrics, save_path)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
