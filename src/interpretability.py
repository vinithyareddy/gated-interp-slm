"""
Interpretability tools for analyzing attention patterns and gate activations
"""
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os


def visualize_attention(
    attention_weights: List[mx.array],
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap
    
    Args:
        attention_weights: List of attention weight arrays per layer
            Each array has shape [batch, n_heads, seq_len, seq_len]
        tokens: List of token strings for labeling
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
        save_path: Optional path to save the figure
    """
    # Extract attention for specific layer and head
    attn = attention_weights[layer_idx]  # [batch, n_heads, seq_len, seq_len]
    attn = np.array(attn[0, head_idx])  # [seq_len, seq_len]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    
    # Labels
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    
    # Colorbar
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_gates(
    gate_scores: List[mx.array],
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize gate activation patterns
    
    Args:
        gate_scores: List of gate score arrays per layer
            Each array has shape [batch, n_heads, seq_len, d_head]
        tokens: List of token strings for labeling
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
        save_path: Optional path to save the figure
    """
    # Extract gates for specific layer and head
    gates = gate_scores[layer_idx]  # [batch, n_heads, seq_len, d_head]
    gates = np.array(gates[0, head_idx])  # [seq_len, d_head]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(gates.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_ylabel('Gate Dimension')
    ax.set_xlabel('Token Position')
    ax.set_title(f'Gate Activations - Layer {layer_idx}, Head {head_idx}')
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Gate Value (0=closed, 1=open)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gate visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_attention_sinks(
    attention_weights: List[mx.array],
    tokens: List[str],
    threshold: float = 0.1
) -> Dict:
    """
    Analyze attention sink behavior
    
    Attention sinks are tokens that receive disproportionate attention
    across many positions (often the first token or special tokens).
    
    Args:
        attention_weights: List of attention weight arrays per layer
        tokens: List of token strings
        threshold: Threshold for considering a token as receiving high attention
        
    Returns:
        analysis: Dictionary with attention sink metrics
    """
    n_layers = len(attention_weights)
    seq_len = len(tokens)
    
    # Aggregate attention across all layers and heads
    total_attention = np.zeros(seq_len)
    
    for layer_attn in attention_weights:
        # layer_attn: [batch, n_heads, seq_len, seq_len]
        layer_attn = np.array(layer_attn[0])  # [n_heads, seq_len, seq_len]
        
        # Sum attention received by each token (across all query positions)
        # This tells us how much total attention each token receives
        attention_received = layer_attn.sum(axis=(0, 1))  # [seq_len]
        total_attention += attention_received
    
    # Normalize
    total_attention = total_attention / total_attention.sum()
    
    # Compute entropy (lower entropy = more concentrated attention = more sinks)
    entropy = -np.sum(total_attention * np.log(total_attention + 1e-10))
    max_entropy = np.log(seq_len)
    normalized_entropy = entropy / max_entropy
    
    # Find tokens receiving disproportionate attention
    mean_attention = 1.0 / seq_len
    sink_indices = np.where(total_attention > mean_attention + threshold)[0]
    
    # Attention concentration (what fraction of attention goes to top 10% of tokens)
    top_k = max(1, seq_len // 10)
    top_k_attention = np.sort(total_attention)[-top_k:].sum()
    
    analysis = {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'attention_distribution': total_attention.tolist(),
        'sink_tokens': [(tokens[i], total_attention[i]) for i in sink_indices],
        'top_k_concentration': top_k_attention,
        'mean_attention': mean_attention
    }
    
    return analysis


def visualize_attention_distribution(
    attention_weights: List[mx.array],
    tokens: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize the distribution of attention across tokens
    
    Args:
        attention_weights: List of attention weight arrays per layer
        tokens: List of token strings
        save_path: Optional path to save the figure
    """
    analysis = analyze_attention_sinks(attention_weights, tokens)
    attention_dist = analysis['attention_distribution']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plot of attention distribution
    bars = ax.bar(range(len(tokens)), attention_dist)
    
    # Highlight attention sinks
    mean_attn = analysis['mean_attention']
    threshold = mean_attn + 0.1
    for i, attn in enumerate(attention_dist):
        if attn > threshold:
            bars[i].set_color('red')
    
    # Add horizontal line for mean attention
    ax.axhline(y=mean_attn, color='blue', linestyle='--', label='Mean attention')
    ax.axhline(y=threshold, color='orange', linestyle='--', label='Sink threshold')
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Total Attention Received')
    ax.set_title(f'Attention Distribution (Entropy: {analysis["normalized_entropy"]:.3f})')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention distribution saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_gate_impact(
    attention_weights: List[mx.array],
    gate_scores: List[mx.array],
    tokens: List[str],
    layer_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Compare attention patterns before and after gating
    
    Args:
        attention_weights: Attention weights (before gating)
        gate_scores: Gate activation scores
        tokens: Token strings
        layer_idx: Layer to analyze
        save_path: Optional save path
    """
    # Get attention and gates for specific layer
    attn = np.array(attention_weights[layer_idx][0])  # [n_heads, seq_len, seq_len]
    gates = np.array(gate_scores[layer_idx][0])  # [n_heads, seq_len, d_head]
    
    # Average gate values across dimensions for each position
    avg_gates = gates.mean(axis=-1)  # [n_heads, seq_len]
    
    # For each head, compute correlation between attention and gates
    n_heads = attn.shape[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average attention per head
    ax = axes[0, 0]
    avg_attn = attn.mean(axis=1)  # [n_heads, seq_len]
    for head in range(min(4, n_heads)):
        ax.plot(avg_attn[head], label=f'Head {head}')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Average Attention')
    ax.set_title('Attention Patterns by Head')
    ax.legend()
    
    # Plot 2: Average gates per head
    ax = axes[0, 1]
    for head in range(min(4, n_heads)):
        ax.plot(avg_gates[head], label=f'Head {head}')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Average Gate Value')
    ax.set_title('Gate Activations by Head')
    ax.legend()
    
    # Plot 3: Gate distribution
    ax = axes[1, 0]
    ax.hist(gates.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Gate Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Gate Values')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    ax.legend()
    
    # Plot 4: Correlation between attention and gates
    ax = axes[1, 1]
    # Compute per-position correlation
    correlations = []
    for head in range(n_heads):
        attn_flat = attn[head].mean(axis=0)  # Average attention received
        gate_flat = avg_gates[head]
        corr = np.corrcoef(attn_flat, gate_flat)[0, 1]
        correlations.append(corr)
    
    ax.bar(range(n_heads), correlations)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation between Attention and Gates')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gate impact analysis saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
