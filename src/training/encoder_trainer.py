"""
Siamese Encoder Trainer - Contrastive Learning - Built from Scratch

Loss Function: Triplet Margin Loss

For each triplet (job, positive_resume, negative_resume):

    loss = max(0, margin + d(job, positive) - d(job, negative))

    Where d = 1 - cosine_similarity (distance)

    In plain English:
        "The job-positive similarity must be greater than
         job-negative similarity by at least `margin`"

    If the model gets it right with margin to spare → loss = 0
    If it gets confused → loss > 0 → gradients update weights

After training:
    - similar job+resume pairs: cosine similarity ~0.8-0.9
    - dissimilar pairs:         cosine similarity ~0.1-0.2
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.encoder import SiameseEncoder
from src.data.matching_dataset import MatchingDataset


def get_device() -> torch.device:
    """M1 GPU (MPS) > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss from scratch

    Pushes positive pairs together and negative pairs apart.
    """

    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin: Minimum required gap between positive and negative distances
                    Higher = stricter separation (0.2-0.5 is typical)
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor:   torch.Tensor,   # job embedding       [batch, dim]
        positive: torch.Tensor,   # matching resume     [batch, dim]
        negative: torch.Tensor,   # non-matching resume [batch, dim]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute triplet loss

        Returns:
            (loss, metrics_dict)
        """
        # Cosine similarity (dot product since L2-normalized)
        sim_pos = (anchor * positive).sum(dim=1)  # [batch] — should be HIGH
        sim_neg = (anchor * negative).sum(dim=1)  # [batch] — should be LOW

        # Convert similarity to distance (0 = identical, 2 = opposite)
        dist_pos = 1.0 - sim_pos
        dist_neg = 1.0 - sim_neg

        # Triplet loss: penalize when positive isn't far enough from negative
        losses = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        loss   = losses.mean()

        # Metrics for monitoring
        metrics = {
            "sim_positive": sim_pos.mean().item(),   # want this HIGH (~0.8+)
            "sim_negative": sim_neg.mean().item(),   # want this LOW  (~0.1-)
            "gap":          (sim_pos - sim_neg).mean().item(),  # want this > margin
            "hard_triplets": (losses > 0).float().mean().item(),  # fraction still learning
        }

        return loss, metrics


class EncoderTrainer:
    """
    Trains the Siamese Encoder using contrastive learning

    Usage:
        trainer = EncoderTrainer(model)
        trainer.train(dataset, epochs=30)
        trainer.save("models/encoder/best_model.pt")
    """

    def __init__(
        self,
        model:           SiameseEncoder,
        learning_rate:   float = 3e-4,
        batch_size:      int   = 16,
        margin:          float = 0.3,
        checkpoint_dir:  str   = "models/encoder/checkpoints",
    ):
        self.model     = model
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Loss
        self.criterion = TripletLoss(margin=margin)

        # Optimizer — lower LR than classifier (transformer is more sensitive)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-2,
        )

        # Warmup + cosine decay schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-5
        )

        # Tracking
        self.best_gap   = -float('inf')
        self.best_epoch = 0

    def train(
        self,
        dataset:   MatchingDataset,
        epochs:    int   = 30,
        val_split: float = 0.2,
        patience:  int   = 8,
    ):
        """Train the encoder"""
        # Split train/val
        val_size   = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=self.batch_size, shuffle=False)

        print(f"Train: {train_size} triplets | Val: {val_size} triplets")
        print("-" * 70)

        no_improve = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_metrics = self._train_epoch(train_loader, epoch, epochs)

            # Validate
            val_loss, val_metrics = self._validate(val_loader)

            self.scheduler.step()

            # Print
            print(
                f"Loss: {train_loss:.4f} | "
                f"sim+ {val_metrics['sim_positive']:.3f} | "
                f"sim- {val_metrics['sim_negative']:.3f} | "
                f"gap: {val_metrics['gap']:.3f} | "
                f"hard: {val_metrics['hard_triplets']:.0%}"
            )

            # Save best (maximize gap between positive and negative similarity)
            gap = val_metrics["gap"]
            if gap > self.best_gap:
                self.best_gap   = gap
                self.best_epoch = epoch
                no_improve      = 0
                self._save_checkpoint("best_model.pt")
                print(f"  ✅ Best model saved (gap={gap:.3f})")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            print()

        print("=" * 70)
        print(f"Training Complete! Best gap: {self.best_gap:.3f} at epoch {self.best_epoch}")
        print("=" * 70)

    def _train_epoch(self, loader: DataLoader, epoch: int, total: int) -> Tuple:
        self.model.train()
        total_loss  = 0.0
        all_metrics = {"sim_positive": 0, "sim_negative": 0, "gap": 0, "hard_triplets": 0}

        print(f"Epoch {epoch}/{total}", end=" | ")

        for batch in loader:
            job      = batch["job"].to(self.device)
            positive = batch["positive"].to(self.device)
            negative = batch["negative"].to(self.device)

            self.optimizer.zero_grad()

            # Encode all three
            job_emb  = self.model.encode(job)
            pos_emb  = self.model.encode(positive)
            neg_emb  = self.model.encode(negative)

            loss, metrics = self.criterion(job_emb, pos_emb, neg_emb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k in all_metrics:
                all_metrics[k] += metrics[k]

        n = max(len(loader), 1)
        avg_metrics = {k: v / n for k, v in all_metrics.items()}
        return total_loss / n, avg_metrics

    def _validate(self, loader: DataLoader) -> Tuple:
        self.model.eval()
        total_loss  = 0.0
        all_metrics = {"sim_positive": 0, "sim_negative": 0, "gap": 0, "hard_triplets": 0}

        with torch.no_grad():
            for batch in loader:
                job      = batch["job"].to(self.device)
                positive = batch["positive"].to(self.device)
                negative = batch["negative"].to(self.device)

                job_emb = self.model.encode(job)
                pos_emb = self.model.encode(positive)
                neg_emb = self.model.encode(negative)

                loss, metrics = self.criterion(job_emb, pos_emb, neg_emb)
                total_loss += loss.item()
                for k in all_metrics:
                    all_metrics[k] += metrics[k]

        n = max(len(loader), 1)
        avg_metrics = {k: v / n for k, v in all_metrics.items()}
        return total_loss / n, avg_metrics

    def _save_checkpoint(self, filename: str):
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_gap":             self.best_gap,
            "best_epoch":           self.best_epoch,
            "model_config": {
                "vocab_size":    self.model.token_embedding.num_embeddings,
                "d_model":       self.model.d_model,
                "embedding_dim": self.model.embedding_dim,
                "num_layers":    len(self.model.layers),
                "max_len":       self.model.pos_encoding.pe.size(1) - 1,
            },
        }, self.checkpoint_dir / filename)

    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
