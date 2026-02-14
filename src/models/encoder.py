"""
Siamese Encoder - Job-Resume Matching Model - Built from Scratch

How it works:
    The SAME encoder processes both job descriptions and resumes.
    Both get compressed into a fixed-size embedding vector.
    Cosine similarity between the two vectors = match score.

    Job Description  ──┐
                       ├── [Same Encoder] ──► job_embedding  ──┐
    Resume           ──┘                                        ├── cosine_similarity → match score
                       ├── [Same Encoder] ──► resume_embedding─┘
                       └── (weights are shared — that's what "Siamese" means)

Training with Contrastive Loss:
    Positive pair: (job, matching_resume)   → push embeddings CLOSER
    Negative pair: (job, irrelevant_resume) → push embeddings FURTHER APART

    The model learns: similar content → similar vectors

Architecture:
    Token IDs
        ↓
    Embedding + Positional Encoding
        ↓
    N × Transformer Encoder Blocks    (self-attention + FFN)
        ↓
    [CLS] token pooling               (one vector for whole sequence)
        ↓
    Linear projection                 (compress to final embedding size)
        ↓
    L2 Normalize                      (unit sphere — cosine sim = dot product)
        ↓
    Embedding vector (e.g. 256-dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.transformer_block import TransformerEncoderBlock


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding

    Adds position information to embeddings since Transformer
    has no built-in sense of token order.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute positional encodings once
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (saved with model, not a trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return self.dropout(x + self.pe[:, :x.size(1)])


class SiameseEncoder(nn.Module):
    """
    Siamese Encoder for Job-Resume Matching

    Encodes any text (job or resume) into a fixed-size embedding.
    Similarity between embeddings = match score.

    Recommended config for M1 Mac:
        d_model=256, num_heads=4, num_layers=4 → ~10M params, fast to train
    """

    def __init__(
        self,
        vocab_size:    int,
        d_model:       int   = 256,
        num_heads:     int   = 4,
        num_layers:    int   = 4,
        d_ff:          int   = 512,
        embedding_dim: int   = 128,
        max_len:       int   = 256,
        dropout:       float = 0.1,
        padding_idx:   int   = 0,
    ):
        """
        Args:
            vocab_size:    Vocabulary size
            d_model:       Internal transformer dimension
            num_heads:     Attention heads (d_model must be divisible by this)
            num_layers:    Number of transformer blocks stacked
            d_ff:          Feed-forward dimension (usually 2× d_model)
            embedding_dim: Final output embedding size
            max_len:       Max sequence length
            dropout:       Dropout rate
            padding_idx:   Index of [PAD] token
        """
        super().__init__()

        self.d_model       = d_model
        self.embedding_dim = embedding_dim

        # ── Token Embedding ────────────────────────────────────────────────
        # +1 for [CLS] token that we prepend to every sequence
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # ── Positional Encoding ────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, max_len + 1, dropout)

        # ── Transformer Encoder Stack ──────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # ── Output Projection ──────────────────────────────────────────────
        # Compress d_model → embedding_dim for efficient similarity computation
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )

        # [CLS] token: learnable vector prepended to every sequence
        # The model learns to compress full-sequence meaning into this token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.token_embedding.weight, -0.1, 0.1)
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _make_padding_mask(
        self,
        token_ids: torch.Tensor,
        has_cls:   bool = True,
    ) -> torch.Tensor:
        """
        Create mask that hides [PAD] tokens from attention

        Args:
            token_ids: [batch, seq_len]
            has_cls:   Whether CLS token is prepended (adds 1 to length)

        Returns:
            [batch, 1, 1, seq_len] — broadcastable mask
        """
        # 1 for real tokens, 0 for padding
        mask = (token_ids != 0).float()  # [batch, seq_len]

        if has_cls:
            # CLS token is always real (never padded)
            cls_mask = torch.ones(token_ids.size(0), 1, device=token_ids.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        return mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len+1]

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs into a fixed-size embedding vector

        Args:
            token_ids: [batch, seq_len] — job description or resume tokens

        Returns:
            [batch, embedding_dim] — L2-normalized embedding
        """
        B, T = token_ids.shape

        # Step 1: Embed tokens and scale
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)  # [B, T, d_model]

        # Step 2: Prepend [CLS] token to every sequence
        cls_tokens = self.cls_token.expand(B, 1, self.d_model)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)                   # [B, T+1, d_model]

        # Step 3: Add positional encoding
        x = self.pos_encoding(x)

        # Step 4: Create padding mask
        mask = self._make_padding_mask(token_ids, has_cls=True)

        # Step 5: Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Step 6: Extract [CLS] token (position 0) as sequence representation
        cls_output = x[:, 0, :]  # [B, d_model]

        # Step 7: Project to final embedding dimension
        embedding = self.projector(cls_output)  # [B, embedding_dim]

        # Step 8: L2 normalize (unit sphere)
        # This makes cosine similarity = dot product, numerically stable
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(
        self,
        job_token_ids:    torch.Tensor,
        resume_token_ids: torch.Tensor,
    ) -> tuple:
        """
        Encode both job and resume, return embeddings

        Args:
            job_token_ids:    [batch, seq_len]
            resume_token_ids: [batch, seq_len]

        Returns:
            (job_embeddings, resume_embeddings) each [batch, embedding_dim]
        """
        job_emb    = self.encode(job_token_ids)
        resume_emb = self.encode(resume_token_ids)
        return job_emb, resume_emb

    def similarity(
        self,
        job_token_ids:    torch.Tensor,
        resume_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between job and resume

        Returns:
            [batch] similarity scores between -1 and 1
            (in practice 0.0 to 1.0 after training)
        """
        job_emb, resume_emb = self.forward(job_token_ids, resume_token_ids)
        # Since embeddings are L2-normalized, dot product = cosine similarity
        return (job_emb * resume_emb).sum(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
