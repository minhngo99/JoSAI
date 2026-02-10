"""
BiLSTM-CRF Named Entity Recognition Model - Built from Scratch

Architecture:
    Input tokens
        ↓
    Embedding Layer     (token indices → dense vectors)
        ↓
    Bidirectional LSTM  (reads sequence forward AND backward)
        ↓
    Linear Projection   (LSTM hidden → label scores per token)
        ↓
    CRF Layer           (finds globally best label sequence)
        ↓
    Output labels

Why BiLSTM?
    - Forward LSTM: sees past context  ("Python ... engineer")
    - Backward LSTM: sees future context ("engineer ... Python")
    - Together: full context for each token

Why CRF on top?
    - Enforces valid label sequences
    - B-SKILL must come before I-SKILL
    - Can't have I-TECH without B-TECH first
"""
import torch
import torch.nn as nn
from .crf import CRFLayer


class BiLSTMCRF(nn.Module):
    """
    Bidirectional LSTM + CRF for Named Entity Recognition

    Extracts entities from job postings:
        - SKILL      (e.g. "problem solving", "communication")
        - TECH       (e.g. "Python", "Docker", "PostgreSQL")
        - REQUIREMENT (e.g. "5+ years experience", "BS degree")
        - LEVEL      (e.g. "Senior", "Junior", "Lead")
        - O          (outside - not an entity)
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size:      Number of tokens in vocabulary
            num_labels:      Number of NER label types
            embedding_dim:   Size of token embeddings (128 is good for small models)
            hidden_dim:      LSTM hidden size (256 per direction = 512 total)
            num_lstm_layers: Number of stacked LSTM layers (2 is standard)
            dropout:         Dropout rate for regularization
            padding_idx:     Index of [PAD] token (to ignore in embeddings)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # ── Layer 1: Token Embeddings ──────────────────────────────────────
        # Convert token index → dense vector
        # padding_idx=0 means [PAD] tokens always get zero embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # ── Layer 2: Dropout on Embeddings ────────────────────────────────
        self.embedding_dropout = nn.Dropout(dropout)

        # ── Layer 3: Bidirectional LSTM ───────────────────────────────────
        # batch_first=True: input shape [batch, seq_len, features]
        # bidirectional=True: reads left→right AND right→left
        # Output size = hidden_dim * 2 (forward + backward)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # ── Layer 4: Dropout on LSTM output ──────────────────────────────
        self.lstm_dropout = nn.Dropout(dropout)

        # ── Layer 5: Linear Projection ────────────────────────────────────
        # Project from LSTM output (hidden_dim * 2) → label scores (num_labels)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        # ── Layer 6: CRF ──────────────────────────────────────────────────
        self.crf = CRFLayer(num_labels)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability"""
        # Embedding: small uniform values
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # LSTM: orthogonal initialization for recurrent weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

        # Linear: Xavier uniform
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass - compute emission scores for each token

        Args:
            token_ids: [batch, seq_len] token indices
            mask:      [batch, seq_len] 1=real token, 0=padding

        Returns:
            emissions: [batch, seq_len, num_labels] label scores per token
        """
        # Step 1: Embed tokens
        # [batch, seq_len] → [batch, seq_len, embedding_dim]
        embedded = self.embedding(token_ids)
        embedded = self.embedding_dropout(embedded)

        # Step 2: Pack padded sequence (speeds up LSTM on padded batches)
        seq_lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # Step 3: Run through BiLSTM
        lstm_out, _ = self.lstm(packed)

        # Unpack back to padded tensor
        # lstm_out: [batch, seq_len, hidden_dim * 2]
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out,
            batch_first=True,
            total_length=token_ids.size(1)
        )
        lstm_out = self.lstm_dropout(lstm_out)

        # Step 4: Project to label scores
        # [batch, seq_len, hidden_dim * 2] → [batch, seq_len, num_labels]
        emissions = self.classifier(lstm_out)

        return emissions

    def loss(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CRF loss for training

        Args:
            token_ids: [batch, seq_len]
            labels:    [batch, seq_len] correct label indices
            mask:      [batch, seq_len]

        Returns:
            Scalar loss value
        """
        emissions = self.forward(token_ids, mask)
        return self.crf.compute_loss(emissions, labels, mask)

    def predict(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> list:
        """
        Predict NER labels using Viterbi decoding

        Args:
            token_ids: [batch, seq_len]
            mask:      [batch, seq_len]

        Returns:
            List of label sequences (one per batch item)
        """
        with torch.no_grad():
            emissions = self.forward(token_ids, mask)
            predictions = self.crf.decode(emissions, mask)
        return predictions

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
