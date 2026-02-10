"""
Conditional Random Field (CRF) Layer - Built from Scratch

CRF is used as the final layer in NER models because:
- It learns transitions between labels (e.g. B-SKILL can follow O, but not I-TECH)
- Produces globally optimal label sequences (not just greedy per-token)
- Much better than softmax for sequence labeling tasks

How it works:
    Instead of predicting each token label independently,
    CRF scores the entire sequence and finds the best global assignment.
"""
import torch
import torch.nn as nn


class CRFLayer(nn.Module):
    """
    Linear-chain Conditional Random Field

    Learns which label transitions are valid/likely.
    For example:
        - O  → B-SKILL  is valid (start a new skill entity)
        - B-SKILL → I-SKILL is valid (continue a skill entity)
        - I-SKILL → B-TECH  is NOT valid (can't start new entity mid-entity)
    """

    def __init__(self, num_labels: int):
        """
        Args:
            num_labels: Number of NER label types
        """
        super().__init__()
        self.num_labels = num_labels

        # Transition matrix: transitions[i][j] = score of going from label i to label j
        # Learned during training
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

        # Special scores for starting/ending sequences
        # START_TAG and END_TAG are virtual labels
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))

        # Initialize transitions
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for training

        Args:
            emissions: [batch, seq_len, num_labels] scores from BiLSTM
            mask:      [batch, seq_len] 1 for real tokens, 0 for padding

        Returns:
            loss: scalar mean NLL loss
        """
        # Score of correct label sequences (from training labels stored in emissions)
        # We pass labels separately via compute_loss()
        return self._compute_normalizer(emissions, mask)

    def compute_loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NLL loss: log(Z) - score(correct sequence)

        Args:
            emissions: [batch, seq_len, num_labels]
            labels:    [batch, seq_len] correct label indices
            mask:      [batch, seq_len]

        Returns:
            Scalar loss
        """
        # Score of correct sequences
        gold_score = self._score_sequence(emissions, labels, mask)

        # Log partition function (normalizer over all sequences)
        forward_score = self._compute_normalizer(emissions, mask)

        # NLL = log(Z) - score(correct)
        loss = forward_score - gold_score

        return loss.mean()

    def _score_sequence(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score of the correct label sequence

        Args:
            emissions: [batch, seq_len, num_labels]
            labels:    [batch, seq_len]
            mask:      [batch, seq_len]

        Returns:
            [batch] scores
        """
        batch_size, seq_len, _ = emissions.shape

        # Start transition scores
        score = self.start_transitions[labels[:, 0]]

        # Add emission score for first token
        score += emissions[:, 0].gather(1, labels[:, 0].unsqueeze(1)).squeeze(1)

        # Add transition + emission scores for each subsequent token
        for t in range(1, seq_len):
            # Transition score: from labels[t-1] to labels[t]
            trans_score = self.transitions[labels[:, t - 1], labels[:, t]]

            # Emission score: score of labels[t] at position t
            emit_score = emissions[:, t].gather(
                1, labels[:, t].unsqueeze(1)
            ).squeeze(1)

            # Only add score for non-padded positions
            score += (trans_score + emit_score) * mask[:, t]

        # End transition scores
        # Find the last real token for each sequence
        last_tags = labels.gather(
            1, (mask.sum(1) - 1).long().unsqueeze(1)
        ).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log partition function using forward algorithm

        This sums over all possible label sequences (log-space for stability)

        Args:
            emissions: [batch, seq_len, num_labels]
            mask:      [batch, seq_len]

        Returns:
            [batch] log partition values
        """
        batch_size, seq_len, num_labels = emissions.shape

        # Initialize: score of all paths starting with each label
        # [batch, num_labels]
        score = self.start_transitions + emissions[:, 0]

        # Forward pass over each time step
        for t in range(1, seq_len):
            # Broadcast for all (from_label, to_label) combinations
            # [batch, num_labels, 1] + [num_labels, num_labels] + [batch, 1, num_labels]
            broadcast_score = score.unsqueeze(2)              # [batch, num_labels, 1]
            broadcast_emit = emissions[:, t].unsqueeze(1)     # [batch, 1, num_labels]

            # Score of transitioning from each label to each label
            next_score = broadcast_score + self.transitions + broadcast_emit  # [batch, num_labels, num_labels]

            # Log-sum-exp over "from" labels
            next_score = torch.logsumexp(next_score, dim=1)   # [batch, num_labels]

            # Only update scores for non-padded positions
            score = torch.where(mask[:, t].unsqueeze(1).bool(), next_score, score)

        # Add end transition scores
        score += self.end_transitions

        # Final log-sum-exp over all labels
        return torch.logsumexp(score, dim=1)  # [batch]

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list:
        """
        Find best label sequence using Viterbi algorithm

        Viterbi finds the globally optimal sequence (best path through the CRF).

        Args:
            emissions: [batch, seq_len, num_labels]
            mask:      [batch, seq_len]

        Returns:
            List of label sequences (one per batch item)
        """
        batch_size, seq_len, num_labels = emissions.shape

        # Initialize Viterbi scores and backpointers
        viterbi_score = self.start_transitions + emissions[:, 0]  # [batch, num_labels]
        backpointers = []

        # Forward pass
        for t in range(1, seq_len):
            broadcast_score = viterbi_score.unsqueeze(2)           # [batch, num_labels, 1]
            broadcast_emit = emissions[:, t].unsqueeze(1)          # [batch, 1, num_labels]

            # Score of all transitions
            trans_score = broadcast_score + self.transitions       # [batch, num_labels, num_labels]

            # Best previous label for each current label
            best_score, best_label = trans_score.max(dim=1)        # [batch, num_labels]

            backpointers.append(best_label)

            # Update Viterbi scores
            viterbi_score = best_score + emissions[:, t]

        # Add end transitions
        viterbi_score += self.end_transitions

        # Find best final label
        best_final_score, best_final_label = viterbi_score.max(dim=1)  # [batch]

        # Backtrack to find best path
        best_paths = []

        for b in range(batch_size):
            seq_length = int(mask[b].sum().item())

            # Start from best final label
            best_path = [best_final_label[b].item()]

            # Backtrack
            for bp in reversed(backpointers[:seq_length - 1]):
                best_path.append(bp[b][best_path[-1]].item())

            # Reverse to get correct order
            best_path.reverse()
            best_paths.append(best_path)

        return best_paths
