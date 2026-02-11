"""Quick test for checkpoint saving"""
import mlx.core as mx
import sys
import os
sys.path.insert(0, '.')

from src.config import ModelConfig, TrainingConfig
from src.model import GatedTransformer
from src.train import save_checkpoint, load_checkpoint
import mlx.optimizers as optim

# Create small model
config = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128)
model = GatedTransformer(config)

# Create optimizer
optimizer = optim.AdamW(learning_rate=1e-4)

# Try to save
save_path = './test_checkpoint.safetensors'
save_checkpoint(model, optimizer, step=10, epoch=1, metrics={'loss': 2.5}, save_path=save_path)

print("✓ Checkpoint saved successfully!")

# Try to load
model2 = GatedTransformer(config)
step, epoch, metrics = load_checkpoint(model2, optimizer, save_path)

print(f"✓ Checkpoint loaded successfully!")
print(f"  Step: {step}, Epoch: {epoch}, Metrics: {metrics}")

# Cleanup
os.remove(save_path)
os.remove(save_path.replace('.safetensors', '_metadata.json'))
print("✓ Test passed!")
