"""
Matching Dataset - Contrastive Learning Pairs

Training Strategy: Contrastive Loss (SimCSE-style)

For each job description, we need:
    - Positive resume: one that matches well
    - Negative resume: one that doesn't match

The model learns:
    - Similar job+resume pairs → embeddings CLOSE together
    - Dissimilar pairs         → embeddings FAR APART

Real-world data collection:
    Positive pairs: Jobs you applied to + your actual resume
    Negative pairs: Same job + a resume for a different field (e.g., marketing)

For now: synthetic pairs based on overlapping skills/keywords
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.tokenization import BPETokenizer


# ── Sample Training Pairs ──────────────────────────────────────────────────────
# Format: (job_text, positive_resume, negative_resume)
# positive = good match, negative = bad match

SAMPLE_PAIRS: List[Tuple[str, str, str]] = [
    (
        # Job
        "Senior Python Engineer Django REST API PostgreSQL Docker AWS microservices 5 years",
        # Positive (matching resume)
        "Software engineer 6 years Python Django REST API PostgreSQL Docker AWS deployed microservices",
        # Negative (non-matching resume)
        "Marketing manager digital campaigns social media SEO content strategy brand management",
    ),
    (
        "Machine Learning Engineer PyTorch NLP deep learning model deployment Python production",
        "ML engineer PyTorch TensorFlow NLP models deployed production Python computer vision",
        "Sales executive B2B enterprise pipeline Salesforce quota revenue account management",
    ),
    (
        "Full Stack Developer React TypeScript Node.js MongoDB GraphQL frontend backend",
        "Full stack developer React TypeScript Node.js MongoDB REST GraphQL 4 years experience",
        "Financial analyst Excel modeling forecasting budget variance reporting",
    ),
    (
        "DevOps Engineer Kubernetes Terraform AWS CI CD infrastructure automation Python",
        "DevOps engineer Kubernetes Terraform AWS GCP CI CD Jenkins infrastructure Python Golang",
        "HR business partner recruiting onboarding employee relations talent management",
    ),
    (
        "Data Engineer Apache Spark Airflow Snowflake ETL pipeline Python SQL big data",
        "Data engineer Python Spark Airflow ETL pipelines Snowflake SQL data warehousing",
        "Graphic designer Adobe Figma brand identity visual design illustration",
    ),
    (
        "Frontend Engineer React Next.js TypeScript TailwindCSS performance web",
        "Frontend engineer React Next.js TypeScript CSS performance optimization accessibility",
        "Operations manager supply chain logistics process improvement lean six sigma",
    ),
    (
        "Site Reliability Engineer Python Go distributed systems monitoring Prometheus Kubernetes",
        "SRE Python Go Kubernetes Prometheus Grafana distributed systems reliability incident",
        "Content writer SEO blog articles copywriting editorial calendar",
    ),
    (
        "Backend Engineer Rust systems programming performance low latency databases",
        "Backend engineer Rust C++ systems programming performance databases low latency",
        "Project manager PMP agile waterfall stakeholder budget delivery",
    ),
    (
        "Android Engineer Kotlin Java mobile development Android SDK MVVM Jetpack",
        "Android engineer Kotlin Java mobile development MVVM Jetpack Compose Android SDK",
        "Physical therapist patient care rehabilitation exercise clinical documentation",
    ),
    (
        "Security Engineer penetration testing cloud security Python vulnerability AWS",
        "Security engineer penetration testing cloud AWS Python vulnerability assessment SAST DAST",
        "Teacher curriculum lesson plans classroom management student assessment grading",
    ),
    (
        "New Grad Software Engineer algorithms data structures Python Java computer science",
        "Computer science graduate Python Java algorithms data structures leetcode internship",
        "Registered nurse patient care hospital EMR clinical documentation triage",
    ),
    (
        "iOS Engineer Swift UIKit SwiftUI Objective C Apple mobile Xcode performance",
        "iOS developer Swift SwiftUI UIKit Xcode Apple developer mobile performance testing",
        "Business analyst requirements process mapping stakeholder UAT Excel reporting",
    ),
]


class MatchingDataset(Dataset):
    """
    Dataset of (job, positive_resume, negative_resume) triplets

    Used for contrastive learning — teaches the encoder what
    "similar" and "dissimilar" means in the job search domain.
    """

    def __init__(
        self,
        pairs:      List[Tuple[str, str, str]],
        tokenizer:  BPETokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            pairs:      List of (job, positive_resume, negative_resume)
            tokenizer:  Trained BPE tokenizer
            max_length: Max sequence length
        """
        self.max_length = max_length
        self.examples   = self._process(pairs, tokenizer)

    def _encode(self, text: str, tokenizer: BPETokenizer) -> List[int]:
        """Encode text to padded token ID list"""
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:self.max_length]
        ids += [0] * (self.max_length - len(ids))
        return ids

    def _process(
        self,
        pairs:     List[Tuple[str, str, str]],
        tokenizer: BPETokenizer,
    ) -> List[Dict]:
        processed = []

        for job_text, pos_resume, neg_resume in pairs:
            processed.append({
                "job":      self._encode(job_text,    tokenizer),
                "positive": self._encode(pos_resume,  tokenizer),
                "negative": self._encode(neg_resume,  tokenizer),
            })

        return processed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return {
            "job":      torch.tensor(item["job"],      dtype=torch.long),
            "positive": torch.tensor(item["positive"], dtype=torch.long),
            "negative": torch.tensor(item["negative"], dtype=torch.long),
        }
