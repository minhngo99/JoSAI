"""
NER Model Trainer - Built from Scratch

Handles:
    - Training loop with CRF loss
    - Validation and evaluation
    - Checkpointing (save best model)
    - M1 MPS GPU acceleration
    - Early stopping
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.ner_model import BiLSTMCRF
from src.data.ner_dataset import NERDataset, collate_fn, ID2LABEL, LABEL2ID


def get_device() -> torch.device:
    """
    Get best available device for M1 Mac

    Priority: MPS (M1 GPU) > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NERTrainer:
    """
    Trains the BiLSTM-CRF NER model

    Usage:
        trainer = NERTrainer(model, tokenizer)
        trainer.train(examples, epochs=20)
        trainer.save("models/ner/best_model.pt")
    """

    def __init__(
        self,
        model: BiLSTMCRF,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        checkpoint_dir: str = "models/ner/checkpoints",
    ):
        self.model = model
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get device (M1 GPU if available)
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Optimizer: Adam with weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Learning rate scheduler: reduce on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',       # Maximize F1 score
            factor=0.5,       # Halve LR when plateau
            patience=3,       # Wait 3 epochs before reducing
            verbose=True,
        )

        # Tracking
        self.train_losses = []
        self.val_f1s = []
        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def train(
        self,
        dataset: NERDataset,
        epochs: int = 30,
        val_split: float = 0.15,
        patience: int = 7,
    ):
        """
        Train the model

        Args:
            dataset:   NERDataset with training examples
            epochs:    Maximum training epochs
            val_split: Fraction of data for validation
            patience:  Stop if no improvement after this many epochs
        """
        print("="*60)
        print("Starting NER Model Training")
        print("="*60)
        print(f"Total examples:    {len(dataset)}")
        print(f"Epochs:            {epochs}")
        print(f"Batch size:        {self.batch_size}")
        print(f"Model parameters:  {self.model.count_parameters():,}")
        print(f"Device:            {self.device}")
        print("="*60)

        # Split train/validation
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        print(f"Train examples: {train_size} | Val examples: {val_size}")
        print()

        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # ── Training ───────────────────────────────────────────────────
            train_loss = self._train_epoch(train_loader, epoch, epochs)

            # ── Validation ─────────────────────────────────────────────────
            val_f1, val_report = self._validate(val_loader)

            # Track history
            self.train_losses.append(train_loss)
            self.val_f1s.append(val_f1)

            # Update learning rate scheduler
            self.scheduler.step(val_f1)

            # Print epoch summary
            print(f"  Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            if val_report:
                print(f"  Per-class F1: {val_report}")

            # ── Save Best Model ─────────────────────────────────────────────
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
                print(f"  ✅ New best model saved (F1={val_f1:.4f})")
            else:
                epochs_without_improvement += 1

            # ── Early Stopping ──────────────────────────────────────────────
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                break

            print()

        print("="*60)
        print("Training Complete!")
        print(f"Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print("="*60)

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Run one training epoch, return average loss"""
        self.model.train()
        total_loss = 0.0

        print(f"Epoch {epoch}/{total_epochs}", end=" | ")

        for batch in loader:
            # Move batch to device
            token_ids = batch["token_ids"].to(self.device)
            label_ids = batch["label_ids"].to(self.device)
            mask      = batch["mask"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward + loss
            loss = self.model.loss(token_ids, label_ids, mask)

            # Backward
            loss.backward()

            # Gradient clipping (prevents exploding gradients in RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _validate(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Run validation, return F1 score"""
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                token_ids = batch["token_ids"].to(self.device)
                label_ids = batch["label_ids"]
                mask      = batch["mask"].to(self.device)

                # Get predictions
                predictions = self.model.predict(token_ids, mask)

                # Collect real labels (only non-padded positions)
                for i, pred in enumerate(predictions):
                    seq_len = len(pred)
                    real_labels = label_ids[i][:seq_len].tolist()
                    all_predictions.extend(pred)
                    all_labels.extend(real_labels)

        # Compute F1
        f1, report = self._compute_f1(all_predictions, all_labels)
        return f1, report

    def _compute_f1(
        self,
        predictions: List[int],
        labels: List[int],
    ) -> Tuple[float, Dict]:
        """
        Compute entity-level F1 score

        Entity F1 is stricter than token F1:
        - The entire entity span must be predicted correctly
        """
        # Count TP, FP, FN per entity type
        entity_counts = {}

        for label_name in ["TECH", "SKILL", "REQUIREMENT", "LEVEL"]:
            entity_counts[label_name] = {"tp": 0, "fp": 0, "fn": 0}

        def extract_entities(label_seq):
            """Extract (start, end, type) tuples from BIO sequence"""
            entities = []
            current_entity = None

            for i, label_id in enumerate(label_seq):
                label = ID2LABEL.get(label_id, "O")

                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = (i, i, entity_type)

                elif label.startswith("I-") and current_entity:
                    entity_type = label[2:]
                    if current_entity[2] == entity_type:
                        current_entity = (current_entity[0], i, entity_type)
                    else:
                        entities.append(current_entity)
                        current_entity = None
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = None

            if current_entity:
                entities.append(current_entity)

            return set(entities)

        pred_entities = extract_entities(predictions)
        gold_entities = extract_entities(labels)

        # Count TP/FP/FN
        for entity in pred_entities:
            entity_type = entity[2]
            if entity_type in entity_counts:
                if entity in gold_entities:
                    entity_counts[entity_type]["tp"] += 1
                else:
                    entity_counts[entity_type]["fp"] += 1

        for entity in gold_entities:
            entity_type = entity[2]
            if entity_type in entity_counts:
                if entity not in pred_entities:
                    entity_counts[entity_type]["fn"] += 1

        # Compute F1 per entity type
        f1_scores = {}
        total_tp = total_fp = total_fn = 0

        for entity_type, counts in entity_counts.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_scores[entity_type] = round(f1, 3)

        # Macro F1
        macro_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0

        return macro_f1, f1_scores

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1":          self.best_val_f1,
            "best_epoch":           self.best_epoch,
            "train_losses":         self.train_losses,
            "val_f1s":              self.val_f1s,
            "model_config": {
                "vocab_size":      self.model.vocab_size,
                "num_labels":      self.model.num_labels,
                "embedding_dim":   self.model.embedding_dim,
                "hidden_dim":      self.model.hidden_dim,
            },
        }, path)

    def load_best_model(self):
        """Load the best saved checkpoint"""
        path = self.checkpoint_dir / "best_model.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model (Val F1={checkpoint['best_val_f1']:.4f})")

    def save(self, filepath: str):
        """Save final model for inference"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
