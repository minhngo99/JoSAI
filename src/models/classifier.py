"""
Text Classifier - Built from Scratch

Architecture: TextCNN (Convolutional Neural Network for Text)

Why CNN for classification (not LSTM)?
    - Much faster to train than LSTM
    - Great at picking up key phrases regardless of position
    - "Senior Python Engineer" → detects "Senior", "Python", "Engineer"
      no matter where they appear in the text
    - Proven to work well for short-to-medium text classification

How it works:
    Input tokens
        ↓
    Embedding Layer        (token indices → dense vectors)
        ↓
    Parallel Conv1D        (3 filter sizes: bigrams, trigrams, 4-grams)
        ↓
    Max-over-time Pooling  (pick strongest signal from each filter)
        ↓
    Concatenate + Dropout
        ↓
    Linear → Sigmoid       (binary: SWE job or not)
        ↓
    Output: probability (0.0 = not SWE, 1.0 = SWE job)

Classes:
    1 = Software Engineering job  (you want this)
    0 = Not a SWE job             (skip this)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text classification

    Detects whether a job posting is a Software Engineering role.

    Uses multiple filter sizes to capture different n-gram patterns:
        - filter_size=2: bigrams  ("Python engineer", "Senior developer")
        - filter_size=3: trigrams ("Senior Python engineer", "machine learning role")
        - filter_size=4: 4-grams  ("5+ years of experience")
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: list = None,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size:    Number of tokens in vocabulary
            embedding_dim: Size of token embeddings
            num_filters:   Number of filters per filter size
            filter_sizes:  List of n-gram sizes to convolve over
            dropout:       Dropout rate (higher = more regularization)
            padding_idx:   Index of [PAD] token
        """
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [2, 3, 4]

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        # ── Layer 1: Token Embeddings ──────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # ── Layer 2: Parallel Convolutional Filters ────────────────────────
        # One Conv1d per filter size — they run in parallel
        # Each conv scans the text looking for patterns of that size
        #
        # Conv1d(in_channels, out_channels, kernel_size):
        #   in_channels  = embedding_dim (features per token)
        #   out_channels = num_filters   (how many patterns to learn)
        #   kernel_size  = filter_size   (n-gram window size)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])

        # ── Layer 3: Dropout ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Layer 4: Output Layer ──────────────────────────────────────────
        # Input: concatenated max-pooled output from all conv filters
        # Output: single logit (binary classification)
        total_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Linear(total_filters, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            token_ids: [batch, seq_len] token indices

        Returns:
            [batch] logits (raw scores, not probabilities)
        """
        # Step 1: Embed tokens
        # [batch, seq_len] → [batch, seq_len, embedding_dim]
        embedded = self.embedding(token_ids)
        embedded = self.dropout(embedded)

        # Step 2: Reshape for Conv1d
        # Conv1d expects [batch, channels, length]
        # [batch, seq_len, embedding_dim] → [batch, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)

        # Step 3: Apply each conv filter + ReLU + max pooling
        pooled_outputs = []

        for conv in self.convs:
            # Convolve: [batch, embedding_dim, seq_len] → [batch, num_filters, seq_len - filter_size + 1]
            conv_out = F.relu(conv(embedded))

            # Max-over-time pooling: take strongest activation across sequence
            # [batch, num_filters, seq_len - filter_size + 1] → [batch, num_filters]
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)

            pooled_outputs.append(pooled)

        # Step 4: Concatenate all pooled outputs
        # [batch, num_filters * len(filter_sizes)]
        cat = torch.cat(pooled_outputs, dim=1)
        cat = self.dropout(cat)

        # Step 5: Classify
        # [batch, total_filters] → [batch, 1] → [batch]
        logits = self.classifier(cat).squeeze(1)

        return logits

    def predict_proba(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get probability of being a SWE job

        Args:
            token_ids: [batch, seq_len]

        Returns:
            [batch] probabilities between 0 and 1
        """
        with torch.no_grad():
            logits = self.forward(token_ids)
            return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
