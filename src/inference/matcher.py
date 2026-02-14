"""
Job-Resume Matcher - Inference

Given your resume and a list of jobs, ranks them by match score.

Output:
    [
        {"title": "Senior Python Engineer", "company": "Stripe",    "score": 0.91},
        {"title": "Backend Engineer",       "company": "Airbnb",    "score": 0.87},
        {"title": "ML Engineer",            "company": "Anthropic", "score": 0.74},
        ...
    ]
"""
import torch
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.encoder import SiameseEncoder
from src.tokenization import BPETokenizer


class JobMatcher:
    """
    Ranks jobs against your resume using the trained Siamese Encoder

    Usage:
        matcher = JobMatcher.from_checkpoint("models/encoder/checkpoints/best_model.pt", tokenizer)

        # Score a single job
        score = matcher.score(resume_text, job_text)
        print(score)  # 0.87

        # Rank a list of jobs
        ranked = matcher.rank(resume_text, jobs, top_k=10)
    """

    def __init__(
        self,
        model:      SiameseEncoder,
        tokenizer:  BPETokenizer,
        max_length: int = 128,
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer: BPETokenizer) -> "JobMatcher":
        """Load matcher from saved checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config     = checkpoint["model_config"]

        model = SiameseEncoder(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
            max_len=config.get("max_len", 128),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Encoder loaded from {checkpoint_path}")
        return cls(model, tokenizer)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize and pad a single text → [1, max_length] tensor"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:self.max_length]
        ids += [0] * (self.max_length - len(ids))
        return torch.tensor([ids], dtype=torch.long).to(self.device)

    def embed(self, text: str) -> torch.Tensor:
        """
        Get embedding vector for any text

        Returns:
            [embedding_dim] tensor (L2 normalized)
        """
        with torch.no_grad():
            token_ids = self._tokenize(text)
            embedding = self.model.encode(token_ids)
        return embedding.squeeze(0).cpu()

    def score(self, resume_text: str, job_text: str) -> float:
        """
        Compute match score between resume and job

        Args:
            resume_text: Your resume as text
            job_text:    Job description text

        Returns:
            Float between 0.0 and 1.0 (higher = better match)
        """
        resume_emb = self.embed(resume_text)
        job_emb    = self.embed(job_text)

        # Cosine similarity (both are L2-normalized, so dot product suffices)
        similarity = (resume_emb * job_emb).sum().item()

        # Scale from [-1,1] to [0,1] for readability
        return round((similarity + 1) / 2, 3)

    def rank(
        self,
        resume_text: str,
        jobs:        List[Dict],
        text_field:  str = "description",
        top_k:       int = None,
    ) -> List[Dict]:
        """
        Rank a list of jobs by match score against your resume

        Args:
            resume_text: Your resume as text
            jobs:        List of job dicts with a text field
            text_field:  Key in job dict containing the description
            top_k:       Return only top K results (None = all)

        Returns:
            Jobs sorted by match score (highest first),
            each job dict has "match_score" added
        """
        # Encode resume once (reused for all jobs)
        resume_emb = self.embed(resume_text)

        scored_jobs = []
        for job in jobs:
            text    = job.get(text_field, "")
            job_emb = self.embed(text)

            similarity = (resume_emb * job_emb).sum().item()
            score      = round((similarity + 1) / 2, 3)

            job_with_score = {**job, "match_score": score}
            scored_jobs.append(job_with_score)

        # Sort by score descending
        scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)

        if top_k is not None:
            scored_jobs = scored_jobs[:top_k]

        return scored_jobs

    def find_similar_jobs(
        self,
        reference_job: str,
        candidate_jobs: List[Dict],
        text_field: str = "description",
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find jobs similar to a reference job

        Useful for: "I liked this job posting, find more like it"
        """
        return self.rank(reference_job, candidate_jobs, text_field, top_k)
