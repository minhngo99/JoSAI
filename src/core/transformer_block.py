"""
Transformer Encoder Block - Built from Scratch

This is the core building block for the matching encoder.
A stack of these layers learns to understand text deeply.

Architecture of one block:
    Input
      ↓
    Multi-Head Self-Attention   (tokens attend to each other)
      ↓
    Add & LayerNorm             (residual connection)
      ↓
    Feed-Forward Network        (per-token transformation)
      ↓
    Add & LayerNorm             (residual connection)
      ↓
    Output

Why Transformer over BiLSTM for matching?
    - BiLSTM processes sequentially (slow, limited context)
    - Transformer attends to ALL tokens simultaneously
    - Better at capturing long-range relationships
    - "Python" in line 1 can directly attend to "5+ years" in line 10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention

    Each token attends to every other token in the sequence.
    Multiple heads let the model attend to different aspects simultaneously:
        Head 1: skill relationships  ("Python" ↔ "Django")
        Head 2: requirement context  ("5+" ↔ "years experience")
        Head 3: role signals         ("Senior" ↔ "Engineer")
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model:   Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout:   Dropout on attention weights
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # Dimension per head

        # Projections for Query, Key, Value + Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [batch, seq_len, d_model]
            mask: [batch, 1, 1, seq_len] — 0 for padding positions

        Returns:
            [batch, seq_len, d_model]
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [B, T, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: [B, T, d_model] → [B, num_heads, T, d_k]
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        # scores: [B, num_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Mask padding positions (set to -inf so softmax → 0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: [B, num_heads, T, d_k]
        context = torch.matmul(attn_weights, V)

        # Merge heads: [B, num_heads, T, d_k] → [B, T, d_model]
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    Applied independently to each token position.
    Expands then compresses: d_model → d_ff → d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer Encoder Block

    Pre-norm variant (more stable training than original paper's post-norm)
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff   = FeedForward(d_model, d_ff, dropout)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [batch, seq_len, d_model]
            mask: padding mask

        Returns:
            [batch, seq_len, d_model]
        """
        # Self-attention with residual + pre-norm
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # Feed-forward with residual + pre-norm
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x
