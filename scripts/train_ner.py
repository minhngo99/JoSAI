"""
Train the NER Model from Scratch

Run this script to train your NER model:

    python scripts/train_ner.py

It will:
    1. Load (or train) the tokenizer
    2. Load training data
    3. Train BiLSTM-CRF from scratch
    4. Save the best model

Typical output:
    Epoch 1/30 | Loss: 3.2104 | Val F1: 0.1230
    Epoch 2/30 | Loss: 2.8853 | Val F1: 0.2410
    ...
    Epoch 15/30 | Loss: 0.4201 | Val F1: 0.7830
    ✅ New best model saved (F1=0.7830)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization import BPETokenizer
from src.models.ner_model import BiLSTMCRF
from src.data.ner_dataset import NERDataset, SAMPLE_DATA, NUM_LABELS
from src.training.ner_trainer import NERTrainer


TOKENIZER_PATH = "data/vocabularies/ner_tokenizer.json"
MODEL_DIR      = "models/ner"
VOCAB_SIZE     = 5000
EMBEDDING_DIM  = 128
HIDDEN_DIM     = 256
LSTM_LAYERS    = 2
DROPOUT        = 0.3
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 8
EPOCHS         = 30


def build_tokenizer(texts):
    """Train tokenizer on the NER corpus"""
    tokenizer_path = Path(TOKENIZER_PATH)

    if tokenizer_path.exists():
        print(f"Loading existing tokenizer from {TOKENIZER_PATH}")
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training new tokenizer...")
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, lowercase=True)
        tokenizer.train(texts)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(TOKENIZER_PATH)

    return tokenizer


def main():
    print("="*60)
    print("NER Model Training — From Scratch")
    print("="*60)

    # ── Step 1: Prepare text corpus for tokenizer ──────────────────────────
    print("\n📚 Step 1: Preparing training data...")
    all_texts = [" ".join(words) for words, _ in SAMPLE_DATA]
    print(f"   {len(SAMPLE_DATA)} annotated examples")

    # ── Step 2: Build / load tokenizer ────────────────────────────────────
    print("\n🔤 Step 2: Tokenizer...")
    tokenizer = build_tokenizer(all_texts)
    print(f"   Vocab size: {len(tokenizer)}")

    # ── Step 3: Create dataset ─────────────────────────────────────────────
    print("\n📦 Step 3: Building dataset...")
    dataset = NERDataset(SAMPLE_DATA, tokenizer, max_length=128)
    print(f"   Dataset size: {len(dataset)} examples")
    print(f"   Label types: {NUM_LABELS}")

    # ── Step 4: Build model ────────────────────────────────────────────────
    print("\n🧠 Step 4: Building model...")
    model = BiLSTMCRF(
        vocab_size=len(tokenizer),
        num_labels=NUM_LABELS,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
    )
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Architecture:")
    print(f"     Embedding:  {len(tokenizer)} × {EMBEDDING_DIM}")
    print(f"     BiLSTM:     {LSTM_LAYERS} layers × {HIDDEN_DIM}×2 hidden")
    print(f"     Classifier: {HIDDEN_DIM*2} → {NUM_LABELS}")
    print(f"     CRF:        {NUM_LABELS}×{NUM_LABELS} transitions")

    # ── Step 5: Train ──────────────────────────────────────────────────────
    print("\n🚀 Step 5: Training...")
    trainer = NERTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        checkpoint_dir=f"{MODEL_DIR}/checkpoints",
    )
    trainer.train(dataset, epochs=EPOCHS, patience=7)

    # ── Step 6: Save final model ───────────────────────────────────────────
    print("\n💾 Step 6: Saving...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save(f"{MODEL_DIR}/best_model.pt")
    print(f"   Saved to {MODEL_DIR}/best_model.pt")
    print(f"   Tokenizer at {TOKENIZER_PATH}")

    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"   Best Val F1: {trainer.best_val_f1:.4f}")
    print(f"   Run demo: python scripts/demo_ner.py")
    print("="*60)


if __name__ == "__main__":
    main()
