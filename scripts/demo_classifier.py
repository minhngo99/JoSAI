"""
Job Classifier Demo — See It Filter Jobs in Real Time

Run AFTER training:
    python scripts/demo_classifier.py

Shows the model classifying raw job postings as SWE or non-SWE,
with confidence scores.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization import BPETokenizer
from src.inference.classifier_predictor import ClassifierPredictor

TOKENIZER_PATH  = "data/vocabularies/classifier_tokenizer.json"
CHECKPOINT_PATH = "models/classifier/checkpoints/best_model.pt"
VOCAB_SIZE      = 5000


# ── Test jobs ──────────────────────────────────────────────────────────────────
TEST_JOBS = [
    # Should be SWE ✅
    ("Senior Python Engineer",
     "We need a Senior Python Engineer with 5+ years experience in Django, PostgreSQL, and Docker. "
     "Strong knowledge of REST APIs and microservices required."),

    ("ML Engineer",
     "Machine Learning Engineer position. Required: PyTorch, TensorFlow, NLP. "
     "Experience deploying ML models to production in AWS."),

    ("Full Stack Developer",
     "Full Stack role: React, TypeScript, Node.js, MongoDB. "
     "3+ years experience. GraphQL and CI/CD knowledge a plus."),

    ("New Grad SWE",
     "New Grad Software Engineer. BS Computer Science. "
     "Strong algorithms and data structures. Python or Java."),

    # Should NOT be SWE ❌
    ("Marketing Manager",
     "Digital Marketing Manager needed. Lead campaigns across social media, "
     "email, and SEO. Manage brand strategy and content calendar."),

    ("Product Manager",
     "Product Manager for our SaaS platform. Drive roadmap, work with engineers "
     "and designers. User research, sprint planning, stakeholder management."),

    ("HR Business Partner",
     "HR Business Partner. Manage recruiting, onboarding, employee relations. "
     "Partner with leadership on talent strategy and compensation."),

    ("Sales Executive",
     "Enterprise Sales Executive. Close B2B deals, manage pipeline, "
     "hit quota. Salesforce CRM experience required."),

    # Edge cases 🤔
    ("Data Analyst",
     "Data Analyst. SQL, Tableau, Excel. Build dashboards and reports. "
     "Business intelligence and stakeholder communication."),

    ("DevOps / SRE",
     "DevOps Engineer. Kubernetes, Terraform, AWS, CI/CD. "
     "Automate infrastructure, improve reliability and deployment pipelines."),
]


def main():
    print("=" * 70)
    print(" JOB CLASSIFIER DEMO ".center(70, "="))
    print("=" * 70)

    # ── Load tokenizer ─────────────────────────────────────────────────────
    if not Path(TOKENIZER_PATH).exists():
        print(f"\n❌ Tokenizer not found. Run: python scripts/train_classifier.py")
        return

    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)

    # ── Load model ─────────────────────────────────────────────────────────
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ Model not found. Run: python scripts/train_classifier.py")
        return

    predictor = ClassifierPredictor.from_checkpoint(CHECKPOINT_PATH, tokenizer)

    # ── Classify jobs ──────────────────────────────────────────────────────
    print(f"\n{'Title':<25} {'Result':<12} {'Confidence':<12} {'Bar'}")
    print("-" * 70)

    swe_count = 0
    for title, description in TEST_JOBS:
        result = predictor.predict(description)

        icon       = "✅" if result["is_swe_job"] else "❌"
        label      = result["label"]
        confidence = result["confidence"]

        # Confidence bar
        bar_len = int(confidence * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"{title:<25} {icon} {label:<10} {confidence:.0%}  {bar}")

        if result["is_swe_job"]:
            swe_count += 1

    print("-" * 70)
    print(f"\nFound {swe_count}/{len(TEST_JOBS)} SWE jobs")

    # ── Show filtering use case ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("How this integrates with your scraper:")
    print("=" * 70)
    print("""
    # In your job scraper pipeline:
    from src.inference.classifier_predictor import ClassifierPredictor

    # Filter scraped jobs — only keep SWE roles
    all_scraped_jobs = scraper.get_jobs()        # e.g., 500 jobs
    swe_jobs = predictor.filter_swe_jobs(        # keeps ~150 SWE jobs
        all_scraped_jobs,
        text_field="description"
    )

    # Now only relevant jobs go to the NER model + matching engine
    """)

    print("=" * 70)
    print("✅ Demo complete! Your classifier is working.")
    print("=" * 70)


if __name__ == "__main__":
    main()
