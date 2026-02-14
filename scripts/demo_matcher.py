"""
Job-Resume Matching Demo — See It Rank Jobs

Run AFTER training:
    python scripts/demo_matcher.py

Shows the encoder ranking a pool of jobs against your resume,
with match scores and a visual bar for each.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization import BPETokenizer
from src.inference.matcher import JobMatcher

TOKENIZER_PATH  = "data/vocabularies/encoder_tokenizer.json"
CHECKPOINT_PATH = "models/encoder/checkpoints/best_model.pt"
VOCAB_SIZE      = 5000


# ── Your resume (in production: loaded from file) ──────────────────────────────
MY_RESUME = """
Software Engineer with 5 years experience.
Python Django FastAPI REST API PostgreSQL Redis Docker AWS Kubernetes.
Built microservices architecture handling 1M+ requests per day.
Strong algorithms and system design. Computer Science degree.
Experience with CI/CD Jenkins GitHub Actions testing.
"""

# ── Pool of jobs to rank ───────────────────────────────────────────────────────
JOB_POOL = [
    {
        "title":   "Senior Python Engineer",
        "company": "Stripe",
        "description": "Senior Python engineer. Django REST API PostgreSQL Redis Docker AWS. "
                       "5+ years required. Microservices and system design experience.",
    },
    {
        "title":   "Backend Engineer",
        "company": "Airbnb",
        "description": "Backend engineer Python FastAPI PostgreSQL Kubernetes AWS. "
                       "3+ years experience. Strong CS fundamentals.",
    },
    {
        "title":   "ML Engineer",
        "company": "Anthropic",
        "description": "Machine learning engineer PyTorch TensorFlow NLP deep learning. "
                       "Production ML systems. Python required.",
    },
    {
        "title":   "Full Stack Developer",
        "company": "Notion",
        "description": "Full stack developer React TypeScript Node.js PostgreSQL. "
                       "4+ years. GraphQL REST APIs.",
    },
    {
        "title":   "DevOps Engineer",
        "company": "Databricks",
        "description": "DevOps engineer Kubernetes Terraform AWS CI/CD infrastructure. "
                       "Python or Go. Strong automation skills.",
    },
    {
        "title":   "iOS Engineer",
        "company": "Linear",
        "description": "iOS engineer Swift SwiftUI UIKit mobile development. "
                       "3+ years iOS experience. Apple developer ecosystem.",
    },
    {
        "title":   "Marketing Manager",
        "company": "Acme Corp",
        "description": "Digital marketing manager. Campaigns SEO social media content. "
                       "Brand strategy and analytics.",
    },
    {
        "title":   "Data Engineer",
        "company": "Databricks",
        "description": "Data engineer Python Spark Airflow Snowflake ETL pipelines SQL. "
                       "Big data experience required.",
    },
    {
        "title":   "New Grad SWE",
        "company": "Google",
        "description": "New grad software engineer. Python Java algorithms data structures. "
                       "Computer science degree. Strong coding skills.",
    },
    {
        "title":   "Sales Manager",
        "company": "Salesforce",
        "description": "Enterprise sales manager B2B pipeline management quota Salesforce CRM. "
                       "Revenue and account growth.",
    },
]


def score_bar(score: float, width: int = 20) -> str:
    """Visual bar for match score"""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def main():
    print("=" * 70)
    print(" JOB-RESUME MATCHER DEMO ".center(70, "="))
    print("=" * 70)

    # ── Load ───────────────────────────────────────────────────────────────
    if not Path(TOKENIZER_PATH).exists() or not Path(CHECKPOINT_PATH).exists():
        print("❌ Run training first: python scripts/train_encoder.py")
        return

    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)

    matcher = JobMatcher.from_checkpoint(CHECKPOINT_PATH, tokenizer)

    # ── Show resume ────────────────────────────────────────────────────────
    print(f"\n📄 Your Resume (excerpt):")
    print(f"   {MY_RESUME.strip()[:120]}...")

    # ── Rank jobs ──────────────────────────────────────────────────────────
    print(f"\n🔍 Ranking {len(JOB_POOL)} jobs...\n")

    ranked = matcher.rank(MY_RESUME, JOB_POOL, top_k=None)

    print(f"{'Rank':<5} {'Score':<8} {'Bar':<22} {'Title':<28} {'Company'}")
    print("-" * 75)

    for i, job in enumerate(ranked, 1):
        score   = job["match_score"]
        bar     = score_bar(score)
        title   = job["title"][:26]
        company = job["company"]

        # Emoji tier
        if score >= 0.75:
            tier = "🔥"
        elif score >= 0.60:
            tier = "✅"
        elif score >= 0.45:
            tier = "🤔"
        else:
            tier = "❌"

        print(f"#{i:<4} {score:.3f}   {bar}  {tier} {title:<26} {company}")

    # ── Summary ────────────────────────────────────────────────────────────
    strong = [j for j in ranked if j["match_score"] >= 0.70]
    weak   = [j for j in ranked if j["match_score"] < 0.45]

    print("\n" + "=" * 70)
    print(f"🔥 Strong matches (≥0.70): {len(strong)} jobs")
    print(f"❌ Weak matches   (<0.45): {len(weak)} jobs")

    if strong:
        print(f"\n   Top pick: {strong[0]['title']} @ {strong[0]['company']} ({strong[0]['match_score']:.3f})")

    print("""
How this fits the pipeline:
    Scraper → Classifier (SWE?) → NER (extract skills) → Matcher (rank) → Alert
                                                                ↑
                                                           You are here
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
