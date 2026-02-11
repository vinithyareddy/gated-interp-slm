# Gated Attention Transformer for Mechanistic Interpretability

A research-grade implementation of a Gated Attention Transformer using Apple's MLX framework, designed to investigate whether gated attention mechanisms reduce "attention sinks" in Small Language Models (SLMs).

## Overview

This project implements a novel **Gated Attention** mechanism where standard attention outputs are modulated by query-dependent sigmoid gates:

```
Output = SDPA(Q, K, V) ⊙ σ(X @ W_gate)
```

where:
- `SDPA` is Scaled Dot-Product Attention
- `σ` is the sigmoid function
- `⊙` denotes element-wise multiplication

## Architecture

### Model Components

- **GatedAttentionHead**: Single attention head with sigmoid gating
- **MultiHeadGatedAttention**: Multi-head variant with per-head gating
- **TransformerBlock**: Standard transformer block with pre-normalization
- **GatedTransformer**: Complete language model with ~25M parameters (default)

### Default Hyperparameters

- Embedding dimension: 512
- Number of layers: 6
- Number of attention heads: 8
- Feedforward dimension: 2048
- Context length: 512 tokens
- Total parameters: ~25M

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd gated-interp-slm

# Install dependencies
pip install -r requirements.txt
```

## Data

This project uses the **TinyStories** dataset, which should be placed in:
- `./data/raw/train.csv`
- `./data/raw/validation.csv`

The dataset is expected to have a `text` column containing the stories.

## Usage

### Quick Start (Testing)

Train on a small subset for quick validation:

```bash
python main.py --max_train_samples 1000 --max_val_samples 100 --epochs 5 --batch_size 8
```

### Full Training

Train on the complete dataset:

```bash
python main.py --epochs 10 --batch_size 32 --save_dir ./results/full_run
```

### Custom Configuration

```bash
python main.py \
  --d_model 512 \
  --n_layers 6 \
  --n_heads 8 \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --epochs 10 \
  --tokenizer_type char \
  --save_dir ./results/custom_run
```

### Resume Training

```bash
python main.py --resume_from ./results/checkpoints/checkpoint_step_5000.safetensors
```

## Interpretability Analysis

The model includes built-in hooks for extracting attention patterns and gate activations:

```python
from src.model import GatedTransformer
from src.interpretability import visualize_attention, visualize_gates, analyze_attention_sinks
import mlx.core as mx

# Load model
model = GatedTransformer(config)
# ... load weights ...

# Get interpretability data
input_ids = mx.array([[...]])  # Your input tokens
logits, interp_data = model(input_ids, return_interpretability=True)

# Analyze attention patterns
attention_weights = interp_data['attention_weights']  # Per-layer attention maps
gate_scores = interp_data['gate_scores']  # Per-layer gate activations

# Visualize
tokens = ["Once", "upon", "a", "time", ...]
visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0)
visualize_gates(gate_scores, tokens, layer_idx=0, head_idx=0)

# Analyze attention sinks
analysis = analyze_attention_sinks(attention_weights, tokens)
print(f"Attention entropy: {analysis['normalized_entropy']:.3f}")
print(f"Sink tokens: {analysis['sink_tokens']}")
```

## Project Structure

```
gated-interp-slm/
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration dataclasses
│   ├── model.py           # Gated Attention Transformer
│   ├── data.py            # Data loading and tokenization
│   ├── train.py           # Training loop
│   └── interpretability.py # Analysis tools
├── data/
│   └── raw/
│       ├── train.csv      # Training data
│       └── validation.csv # Validation data
└── results/
    └── checkpoints/       # Saved models
```

## Key Features

### 1. Gated Attention Mechanism
- Query-dependent sigmoid gates modulate attention outputs
- Enables investigation of attention sink reduction
- Interpretability hooks for gate analysis

### 2. MLX Optimization
- Leverages Apple Silicon unified memory
- Efficient training on M1/M2/M3 chips
- Native support for Apple GPU acceleration

### 3. Interpretability Tools
- Attention heatmap visualization
- Gate activation analysis
- Attention sink detection and quantification
- Entropy-based metrics

### 4. Flexible Tokenization
- Character-level tokenizer (default, simple)
- TikToken BPE tokenizer (optional, standard)

## Training Details

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Cosine decay with linear warmup
- **Gradient Clipping**: Enabled (max norm = 1.0)

### Checkpointing
- Periodic checkpoints every 1000 steps
- Best model saved based on validation loss
- Resumable training from any checkpoint

### Monitoring
- Training loss and perplexity
- Validation metrics every 500 steps
- Learning rate tracking
- Gradient norm monitoring

## Research Goals

This implementation is designed to investigate:

1. **Attention Sinks**: Do gated attention mechanisms reduce the tendency of models to allocate excessive attention to specific tokens (e.g., first token, special tokens)?

2. **Gate Behavior**: How do gates modulate attention across different layers and heads?

3. **Interpretability**: Can gate activations provide insights into what the model is "focusing on"?

## Performance

Expected training time on Apple Silicon:
- **M1/M2 (8GB)**: ~2-4 hours per epoch (full dataset, batch_size=16)
- **M2/M3 Pro (16GB+)**: ~1-2 hours per epoch (full dataset, batch_size=32)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gated_attention_transformer,
  title={Gated Attention Transformer for Mechanistic Interpretability},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/gated-interp-slm}
}
```

## License

MIT License

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
