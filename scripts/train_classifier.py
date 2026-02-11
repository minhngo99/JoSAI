"""
Train Job Classifier from Scratch

Usage:
    python scripts/train_classifier.py

Trains TextCNN to classify SWE vs non-SWE job postings.
Typically converges in < 20 minutes on M1 Mac.

Expected output:
    Epoch 1/20  | Loss: 0.6821 | Val Acc: 62.5% | P: 0.60 R: 0.58
    Epoch 5/20  | Loss: 0.4102 | Val Acc: 80.0% | P: 0.83 R: 0.78
    Epoch 12/20 | Loss: 0.1803 | Val Acc: 92.5% | P: 0.94 R: 0.91
    ✅ New best saved (Acc=92.5%)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization import BPETokenizer
from src.models.classifier import TextCNN
from src.data.classifier_dataset import ClassifierDataset, SAMPLE_DATA
from src.training.classifier_trainer import ClassifierTrainer

# ── Config ─────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "data/vocabularies/classifier_tokenizer.json"
MODEL_DIR      = "models/classifier"
VOCAB_SIZE     = 5000
EMBEDDING_DIM  = 128
NUM_FILTERS    = 128
FILTER_SIZES   = [2, 3, 4]
DROPOUT        = 0.5
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 16
EPOCHS         = 20
MAX_LENGTH     = 128


def build_tokenizer(texts):
    """Train or load tokenizer"""
    path = Path(TOKENIZER_PATH)

    if path.exists():
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer...")
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, lowercase=True)
        tokenizer.train(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(TOKENIZER_PATH)

    return tokenizer


def main():
    print("=" * 60)
    print("Job Classifier Training — From Scratch")
    print("=" * 60)

    # ── Step 1: Prepare corpus ─────────────────────────────────────────────
    print("\n📚 Step 1: Preparing data...")
    texts  = [text for text, _ in SAMPLE_DATA]
    labels = [label for _, label in SAMPLE_DATA]
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"   {len(SAMPLE_DATA)} examples  ({pos} SWE jobs, {neg} non-SWE)")

    # ── Step 2: Tokenizer ──────────────────────────────────────────────────
    print("\n🔤 Step 2: Tokenizer...")
    tokenizer = build_tokenizer(texts)
    print(f"   Vocab size: {len(tokenizer)}")

    # ── Step 3: Dataset ────────────────────────────────────────────────────
    print("\n📦 Step 3: Building dataset...")
    dataset = ClassifierDataset(SAMPLE_DATA, tokenizer, max_length=MAX_LENGTH)
    print(f"   Dataset: {len(dataset)} examples, max_length={MAX_LENGTH}")

    # ── Step 4: Model ──────────────────────────────────────────────────────
    print("\n🧠 Step 4: Building model...")
    model = TextCNN(
        vocab_size=len(tokenizer),
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=DROPOUT,
    )
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Architecture:")
    print(f"     Embedding:  {len(tokenizer)} × {EMBEDDING_DIM}")
    print(f"     Conv filters: {FILTER_SIZES} × {NUM_FILTERS} each")
    print(f"     Classifier: {NUM_FILTERS * len(FILTER_SIZES)} → 1")

    # ── Step 5: Train ──────────────────────────────────────────────────────
    print("\n🚀 Step 5: Training...")
    trainer = ClassifierTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        checkpoint_dir=f"{MODEL_DIR}/checkpoints",
    )
    trainer.train(dataset, epochs=EPOCHS, patience=7)

    # ── Step 6: Save ───────────────────────────────────────────────────────
    print("\n💾 Step 6: Saving...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save(f"{MODEL_DIR}/best_model.pt")
    print(f"   Model saved to {MODEL_DIR}/best_model.pt")
    print(f"   Tokenizer at {TOKENIZER_PATH}")

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Best Val Accuracy: {trainer.best_val_accuracy:.1%}")
    print("   Run demo: python scripts/demo_classifier.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
