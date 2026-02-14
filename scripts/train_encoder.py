"""
Train Job-Resume Matching Encoder from Scratch

Usage:
    python scripts/train_encoder.py

Expected output:
    Epoch 1/30  | Loss: 0.3210 | sim+ 0.321 | sim- 0.298 | gap: 0.023 | hard: 89%
    Epoch 5/30  | Loss: 0.2104 | sim+ 0.612 | sim- 0.201 | gap: 0.411 | hard: 45%
    Epoch 15/30 | Loss: 0.0803 | sim+ 0.841 | sim- 0.134 | gap: 0.707 | hard: 12%
    ✅ Best model saved (gap=0.707)

Key metrics to watch:
    sim+  → similarity of matching pairs  (want > 0.8)
    sim-  → similarity of non-matching    (want < 0.2)
    gap   → sim+ minus sim-               (want > 0.5)
    hard  → fraction of triplets still learning (decreases as model improves)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization import BPETokenizer
from src.models.encoder import SiameseEncoder
from src.data.matching_dataset import MatchingDataset, SAMPLE_PAIRS
from src.training.encoder_trainer import EncoderTrainer

# ── Config ─────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "data/vocabularies/encoder_tokenizer.json"
MODEL_DIR      = "models/encoder"
VOCAB_SIZE     = 5000
D_MODEL        = 256    # Transformer internal dimension
NUM_HEADS      = 4      # Attention heads (D_MODEL must be divisible)
NUM_LAYERS     = 4      # Transformer blocks stacked
D_FF           = 512    # Feed-forward dimension
EMBEDDING_DIM  = 128    # Final embedding size
MAX_LEN        = 128    # Max sequence length
DROPOUT        = 0.1
LEARNING_RATE  = 3e-4
BATCH_SIZE     = 8
EPOCHS         = 30
MARGIN         = 0.3    # Triplet loss margin


def build_tokenizer(texts):
    """Train or load tokenizer"""
    path = Path(TOKENIZER_PATH)
    if path.exists():
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        tok = BPETokenizer(vocab_size=VOCAB_SIZE)
        tok.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer...")
        tok = BPETokenizer(vocab_size=VOCAB_SIZE, lowercase=True)
        tok.train(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(TOKENIZER_PATH)
    return tok


def main():
    print("=" * 70)
    print("Job-Resume Matching Encoder Training — From Scratch")
    print("=" * 70)

    # ── Step 1: Prepare corpus ─────────────────────────────────────────────
    print("\n📚 Step 1: Preparing data...")
    all_texts = []
    for job, pos, neg in SAMPLE_PAIRS:
        all_texts.extend([job, pos, neg])
    print(f"   {len(SAMPLE_PAIRS)} triplets ({len(all_texts)} total texts)")

    # ── Step 2: Tokenizer ──────────────────────────────────────────────────
    print("\n🔤 Step 2: Tokenizer...")
    tokenizer = build_tokenizer(all_texts)
    print(f"   Vocab size: {len(tokenizer)}")

    # ── Step 3: Dataset ────────────────────────────────────────────────────
    print("\n📦 Step 3: Building dataset...")
    dataset = MatchingDataset(SAMPLE_PAIRS, tokenizer, max_length=MAX_LEN)
    print(f"   {len(dataset)} triplets (job, positive, negative)")

    # ── Step 4: Model ──────────────────────────────────────────────────────
    print("\n🧠 Step 4: Building model...")
    model = SiameseEncoder(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    )
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Architecture:")
    print(f"     Embedding:     {len(tokenizer)} × {D_MODEL}")
    print(f"     Transformer:   {NUM_LAYERS} blocks × {NUM_HEADS} heads × {D_MODEL}d")
    print(f"     Output:        {D_MODEL} → {EMBEDDING_DIM}d (L2 normalized)")

    # ── Step 5: Train ──────────────────────────────────────────────────────
    print("\n🚀 Step 5: Training with Triplet Contrastive Loss...")
    print("   sim+ = similarity of matching pairs (want → 1.0)")
    print("   sim- = similarity of non-matching   (want → 0.0)")
    print("   gap  = sim+ - sim-                  (want > 0.5)")
    print()
    trainer = EncoderTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        margin=MARGIN,
        checkpoint_dir=f"{MODEL_DIR}/checkpoints",
    )
    trainer.train(dataset, epochs=EPOCHS, patience=8)

    # ── Step 6: Save ───────────────────────────────────────────────────────
    print("\n💾 Step 6: Saving...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save(f"{MODEL_DIR}/best_model.pt")
    print(f"   Model at {MODEL_DIR}/best_model.pt")
    print(f"   Tokenizer at {TOKENIZER_PATH}")

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"   Best gap: {trainer.best_gap:.3f}")
    print("   Run demo: python scripts/demo_matcher.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
