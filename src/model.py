"""
Gated Attention Transformer Model using MLX

Implements a transformer with gated attention mechanism where attention outputs
are modulated by query-dependent sigmoid gates to investigate attention sinks.
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict, List
import math


class GatedAttentionHead(nn.Module):
    """
    Single attention head with sigmoid gating mechanism.
    
    Formula: Output = SDPA(Q, K, V) ⊙ σ(X @ W_gate)
    where SDPA is Scaled Dot-Product Attention and σ is sigmoid.
    """
    
    def __init__(self, d_model: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        
        # Gate projection (query-dependent)
        self.gate_proj = nn.Linear(d_model, d_head, bias=True)
        
        self.dropout = nn.Dropout(dropout)
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        return_interpretability: bool = False
    ) -> Tuple[mx.array, Optional[Dict]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len] or None
            return_interpretability: Whether to return attention weights and gate scores
            
        Returns:
            output: Gated attention output [batch, seq_len, d_head]
            interp_data: Optional dict with 'attention_weights' and 'gate_scores'
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, d_head]
        K = self.k_proj(x)  # [batch, seq_len, d_head]
        V = self.v_proj(x)  # [batch, seq_len, d_head]
        
        # Scaled dot-product attention
        # Attention scores: Q @ K^T / sqrt(d_head)
        scores = (Q @ K.transpose(0, 2, 1)) * self.scale  # [batch, seq_len, seq_len]
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask
        
        # Attention weights
        attn_weights = mx.softmax(scores, axis=-1)  # [batch, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Attention output
        attn_output = attn_weights @ V  # [batch, seq_len, d_head]
        
        # Compute gate scores (query-dependent sigmoid)
        gate_scores = mx.sigmoid(self.gate_proj(x))  # [batch, seq_len, d_head]
        
        # Apply gating: element-wise multiplication
        gated_output = attn_output * gate_scores  # [batch, seq_len, d_head]
        
        # Prepare interpretability data
        interp_data = None
        if return_interpretability:
            interp_data = {
                'attention_weights': attn_weights,  # [batch, seq_len, seq_len]
                'gate_scores': gate_scores,  # [batch, seq_len, d_head]
            }
        
        return gated_output, interp_data


class MultiHeadGatedAttention(nn.Module):
    """Multi-head gated attention layer"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Create attention heads
        self.heads = [
            GatedAttentionHead(d_model, self.d_head, dropout)
            for _ in range(n_heads)
        ]
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        return_interpretability: bool = False
    ) -> Tuple[mx.array, Optional[Dict]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask
            return_interpretability: Whether to return interpretability data
            
        Returns:
            output: Multi-head attention output [batch, seq_len, d_model]
            interp_data: Optional dict with per-head attention and gate data
        """
        # Run all heads
        head_outputs = []
        all_attn_weights = []
        all_gate_scores = []
        
        for head in self.heads:
            head_out, interp = head(x, mask, return_interpretability)
            head_outputs.append(head_out)
            
            if return_interpretability and interp is not None:
                all_attn_weights.append(interp['attention_weights'])
                all_gate_scores.append(interp['gate_scores'])
        
        # Concatenate heads
        concat_output = mx.concatenate(head_outputs, axis=-1)  # [batch, seq_len, d_model]
        
        # Output projection
        output = self.out_proj(concat_output)
        output = self.dropout(output)
        
        # Aggregate interpretability data
        interp_data = None
        if return_interpretability:
            interp_data = {
                'attention_weights': mx.stack(all_attn_weights, axis=1),  # [batch, n_heads, seq_len, seq_len]
                'gate_scores': mx.stack(all_gate_scores, axis=1),  # [batch, n_heads, seq_len, d_head]
            }
        
        return output, interp_data


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with gated attention and feed-forward network"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadGatedAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        return_interpretability: bool = False
    ) -> Tuple[mx.array, Optional[Dict]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask
            return_interpretability: Whether to return interpretability data
            
        Returns:
            output: Block output [batch, seq_len, d_model]
            interp_data: Optional interpretability data from attention
        """
        # Pre-norm architecture
        # Attention with residual
        normed = self.norm1(x)
        attn_out, interp_data = self.attention(normed, mask, return_interpretability)
        x = x + attn_out
        
        # Feed-forward with residual
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + ff_out
        
        return x, interp_data


class GatedTransformer(nn.Module):
    """
    Complete Gated Attention Transformer for language modeling.
    
    This model uses gated attention to investigate whether gating reduces
    attention sinks in small language models.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ]
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def __call__(
        self, 
        input_ids: mx.array,
        return_interpretability: bool = False
    ) -> Tuple[mx.array, Optional[Dict]]:
        """
        Args:
            input_ids: Token indices [batch, seq_len]
            return_interpretability: Whether to return attention maps and gate scores
            
        Returns:
            logits: Next-token prediction logits [batch, seq_len, vocab_size]
            interp_data: Optional dict with 'attention_weights' and 'gate_scores' per layer
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = mx.arange(seq_len)
        positions = mx.broadcast_to(positions[None, :], (batch_size, seq_len))
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        pos_emb = self.position_embedding(positions)  # [batch, seq_len, d_model]
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask (prevent attending to future tokens)
        mask = self._create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        all_layer_interp = []
        for block in self.blocks:
            x, interp_data = block(x, mask, return_interpretability)
            if return_interpretability and interp_data is not None:
                all_layer_interp.append(interp_data)
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # Aggregate interpretability data
        interp_output = None
        if return_interpretability and all_layer_interp:
            interp_output = {
                'attention_weights': [layer['attention_weights'] for layer in all_layer_interp],
                'gate_scores': [layer['gate_scores'] for layer in all_layer_interp],
            }
        
        return logits, interp_output
    
    def _create_causal_mask(self, seq_len: int) -> mx.array:
        """Create causal mask to prevent attending to future positions"""
        # Create upper triangular matrix of -inf
        mask = mx.full((seq_len, seq_len), -1e9)
        mask = mx.triu(mask, k=1)  # Keep upper triangle (excluding diagonal)
        return mask
    
    def get_num_params(self) -> int:
        """Count total number of parameters"""
        def count_params(params):
            total = 0
            if isinstance(params, dict):
                for v in params.values():
                    total += count_params(v)
            elif isinstance(params, list):
                for v in params:
                    total += count_params(v)
            else:
                # It's an array
                total += params.size
            return total
        
        return count_params(self.parameters())
    
    def generate(
        self, 
        input_ids: mx.array, 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            
        Returns:
            generated: Generated token sequence [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids[:, -self.config.max_seq_length:]
            
            # Forward pass
            logits, _ = self(idx_cond, return_interpretability=False)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # [batch, vocab_size]
            
            # Optional top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = mx.topk(logits, top_k, axis=-1)
                # Set all non-top-k logits to -inf
                logits = mx.full_like(logits, -1e9)
                logits = mx.scatter(logits, top_k_indices, top_k_logits, axis=-1)
            
            # Sample from distribution
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs), axis=-1)  # [batch, 1]
            
            # Append to sequence
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)
        
        return input_ids
