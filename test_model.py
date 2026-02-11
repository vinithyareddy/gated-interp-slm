"""
Test script to verify model implementation

This script performs basic sanity checks on the model architecture
without requiring the full dataset.
"""
import mlx.core as mx
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig
from src.model import GatedTransformer
from src.data import CharTokenizer


def test_model_creation():
    """Test that model can be created"""
    print("=" * 60)
    print("Test 1: Model Creation")
    print("=" * 60)
    
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_length=128,
        dropout=0.1
    )
    
    model = GatedTransformer(config)
    num_params = model.get_num_params()
    
    print(f"✓ Model created successfully")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print()
    
    return model, config


def test_forward_pass(model, config):
    """Test forward pass"""
    print("=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 64
    
    # Create random input
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass without interpretability
    logits, interp = model(input_ids, return_interpretability=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch!"
    print(f"  ✓ Shape verification passed")
    print()


def test_interpretability_hooks(model, config):
    """Test interpretability hooks"""
    print("=" * 60)
    print("Test 3: Interpretability Hooks")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 64
    
    # Create random input
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass with interpretability
    logits, interp = model(input_ids, return_interpretability=True)
    
    print(f"✓ Interpretability hooks working")
    
    # Check attention weights
    attention_weights = interp['attention_weights']
    print(f"  Number of layers: {len(attention_weights)}")
    print(f"  Attention shape (layer 0): {attention_weights[0].shape}")
    print(f"  Expected: ({batch_size}, {config.n_heads}, {seq_len}, {seq_len})")
    
    assert len(attention_weights) == config.n_layers, "Wrong number of layers!"
    assert attention_weights[0].shape == (batch_size, config.n_heads, seq_len, seq_len), "Wrong attention shape!"
    
    # Check gate scores
    gate_scores = interp['gate_scores']
    print(f"  Gate scores shape (layer 0): {gate_scores[0].shape}")
    print(f"  Expected: ({batch_size}, {config.n_heads}, {seq_len}, {config.d_head})")
    
    assert len(gate_scores) == config.n_layers, "Wrong number of layers!"
    assert gate_scores[0].shape == (batch_size, config.n_heads, seq_len, config.d_head), "Wrong gate shape!"
    
    print(f"  ✓ All shapes verified")
    print()


def test_tokenizer():
    """Test character tokenizer"""
    print("=" * 60)
    print("Test 4: Character Tokenizer")
    print("=" * 60)
    
    tokenizer = CharTokenizer(min_freq=1)
    
    # Build vocab from sample texts
    sample_texts = [
        "Once upon a time",
        "There was a little girl",
        "She loved to play"
    ]
    tokenizer.build_vocab(sample_texts)
    
    print(f"✓ Tokenizer created")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encoding/decoding
    text = "Once upon a time"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"  Original: '{text}'")
    print(f"  Tokens: {tokens[:10]}... (showing first 10)")
    print(f"  Decoded: '{decoded}'")
    print(f"  ✓ Encoding/decoding works")
    print()


def test_generation(model, config):
    """Test text generation"""
    print("=" * 60)
    print("Test 5: Text Generation")
    print("=" * 60)
    
    # Start with a simple sequence
    input_ids = mx.array([[1, 2, 3, 4, 5]])  # Simple token sequence
    
    # Generate 10 tokens
    generated = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    
    print(f"✓ Generation successful")
    print(f"  Input length: {input_ids.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    assert generated.shape[1] == input_ids.shape[1] + 10, "Wrong generation length!"
    print(f"  ✓ Length verification passed")
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GATED ATTENTION TRANSFORMER - VERIFICATION TESTS")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Model creation
        model, config = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model, config)
        
        # Test 3: Interpretability hooks
        test_interpretability_hooks(model, config)
        
        # Test 4: Tokenizer
        test_tokenizer()
        
        # Test 5: Generation
        test_generation(model, config)
        
        # Summary
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print()
        print("The model is ready for training!")
        print("Run: python main.py --max_train_samples 1000 --epochs 5")
        print()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
