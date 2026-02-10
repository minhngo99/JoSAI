"""
NER Demo — See Your Trained Model Extract Entities

Run AFTER training:

    python scripts/demo_ner.py

Shows the model extracting Technologies, Skills, Requirements
and Seniority Levels from raw job description text.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization import BPETokenizer
from src.models.ner_model import BiLSTMCRF
from src.data.ner_dataset import NUM_LABELS
from src.inference.ner_predictor import NERPredictor


TOKENIZER_PATH  = "data/vocabularies/ner_tokenizer.json"
CHECKPOINT_PATH = "models/ner/checkpoints/best_model.pt"
VOCAB_SIZE      = 5000


# ── Sample job descriptions to test on ────────────────────────────────────────
JOB_DESCRIPTIONS = [
    "We are hiring a Senior Python Engineer with 5+ years of experience. "
    "Must know Django, PostgreSQL, and Docker. Strong problem solving skills required.",

    "Backend Developer needed. Requirements: 3+ years Python, FastAPI, Redis, AWS. "
    "Good communication and teamwork skills. Junior to mid-level candidates welcome.",

    "Machine Learning Engineer position. Required: PyTorch, TensorFlow, scikit-learn. "
    "Experience with NLP and deploying ML models. Master's degree preferred.",

    "Lead Full Stack Engineer. Must have: React, TypeScript, Node.js, MongoDB, GraphQL. "
    "Leadership experience required. 7+ years total experience.",

    "New Grad Software Engineer. BS in Computer Science. "
    "Strong algorithms and data structures knowledge. Python or Java.",
]


def print_entities(entities: dict):
    """Print extracted entities in a readable format"""
    icons = {
        "technologies": "⚙️  Technologies",
        "skills":       "💡 Skills",
        "requirements": "📋 Requirements",
        "levels":       "🎯 Level",
    }
    for key, icon in icons.items():
        items = entities[key]
        if items:
            print(f"   {icon}: {', '.join(items)}")


def main():
    print("="*70)
    print(" NER DEMO — Entity Extraction from Job Postings ".center(70, "="))
    print("="*70)

    # ── Load tokenizer ─────────────────────────────────────────────────────
    if not Path(TOKENIZER_PATH).exists():
        print(f"\n❌ Tokenizer not found at {TOKENIZER_PATH}")
        print("   Run training first: python scripts/train_ner.py")
        return

    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)

    # ── Load model ─────────────────────────────────────────────────────────
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ Model checkpoint not found at {CHECKPOINT_PATH}")
        print("   Run training first: python scripts/train_ner.py")
        return

    print("Loading model...")
    predictor = NERPredictor.from_checkpoint(CHECKPOINT_PATH, tokenizer)

    # ── Run predictions ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Extracting entities from job descriptions...")
    print("="*70)

    for i, job_text in enumerate(JOB_DESCRIPTIONS, 1):
        print(f"\n📄 Job {i}:")
        print(f"   {job_text[:100]}...")
        print()

        entities = predictor.extract(job_text)
        print_entities(entities)
        print()

    print("="*70)
    print("✅ Demo complete!")
    print("\n💡 Your NER model extracts structured data from raw job text.")
    print("   This feeds directly into your job matching model (next step).")
    print("="*70)


if __name__ == "__main__":
    main()
