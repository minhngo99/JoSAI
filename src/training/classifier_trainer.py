"""
Job Classifier Trainer - Built from Scratch

Trains TextCNN to classify SWE vs non-SWE job postings.
Much faster to train than NER — typically converges in < 20 epochs.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.classifier import TextCNN
from src.data.classifier_dataset import ClassifierDataset


def get_device() -> torch.device:
    """M1 GPU (MPS) > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ClassifierTrainer:
    """
    Trains the TextCNN job classifier

    Usage:
        trainer = ClassifierTrainer(model)
        trainer.train(dataset, epochs=20)
        trainer.save("models/classifier/best_model.pt")
    """

    def __init__(
        self,
        model: TextCNN,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        checkpoint_dir: str = "models/classifier/checkpoints",
    ):
        self.model = model
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device (M1 GPU if available)
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Binary cross-entropy loss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Reduce LR when validation accuracy plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )

        # Tracking
        self.train_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def train(
        self,
        dataset: ClassifierDataset,
        epochs: int = 20,
        val_split: float = 0.2,
        patience: int = 7,
    ):
        """
        Train the model

        Args:
            dataset:   ClassifierDataset
            epochs:    Max training epochs
            val_split: Fraction for validation
            patience:  Early stopping patience
        """
        # Split into train / validation
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

        print(f"Train: {train_size} | Val: {val_size}")
        print("-" * 60)

        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_epoch(train_loader, epoch, epochs)

            # Validate
            val_accuracy, val_precision, val_recall = self._validate(val_loader)

            # Track
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)

            # Print metrics
            print(
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_accuracy:.1%} | "
                f"P: {val_precision:.2f} R: {val_recall:.2f}"
            )

            # LR scheduler
            self.scheduler.step(val_accuracy)

            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
                print(f"  ✅ New best saved (Acc={val_accuracy:.1%})")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            print()

        print("=" * 60)
        print(f"Training Complete! Best Acc: {self.best_val_accuracy:.1%} at epoch {self.best_epoch}")
        print("=" * 60)

    def _train_epoch(self, loader: DataLoader, epoch: int, total: int) -> float:
        """One training epoch"""
        self.model.train()
        total_loss = 0.0

        print(f"Epoch {epoch}/{total}", end=" | ")

        for batch in loader:
            token_ids = batch["token_ids"].to(self.device)
            labels    = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(token_ids)
            loss   = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _validate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Validate — returns (accuracy, precision, recall)"""
        self.model.eval()

        tp = fp = tn = fn = 0

        with torch.no_grad():
            for batch in loader:
                token_ids = batch["token_ids"].to(self.device)
                labels    = batch["label"].cpu()

                probs = self.model.predict_proba(token_ids).cpu()
                preds = (probs >= 0.5).float()

                for pred, label in zip(preds, labels):
                    if pred == 1 and label == 1:
                        tp += 1
                    elif pred == 1 and label == 0:
                        fp += 1
                    elif pred == 0 and label == 0:
                        tn += 1
                    else:
                        fn += 1

        total     = tp + fp + tn + fn
        accuracy  = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return accuracy, precision, recall

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy":    self.best_val_accuracy,
            "best_epoch":           self.best_epoch,
            "model_config": {
                "vocab_size":    self.model.embedding.num_embeddings,
                "embedding_dim": self.model.embedding_dim,
                "num_filters":   self.model.num_filters,
                "filter_sizes":  self.model.filter_sizes,
            },
        }, self.checkpoint_dir / filename)

    def load_best_model(self):
        """Load best checkpoint"""
        path = self.checkpoint_dir / "best_model.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model (Acc={checkpoint['best_val_accuracy']:.1%})")

    def save(self, filepath: str):
        """Save final model for inference"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
