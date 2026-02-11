"""
Job Classifier Predictor - Inference

Given a raw job posting, returns:
    {
        "is_swe_job": True,
        "confidence": 0.94,
        "label": "SWE Job"
    }
"""
import torch
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.classifier import TextCNN
from src.tokenization import BPETokenizer


class ClassifierPredictor:
    """
    Runs inference with the trained job classifier

    Usage:
        predictor = ClassifierPredictor.from_checkpoint("models/classifier/checkpoints/best_model.pt", tokenizer)
        result = predictor.predict("Senior Python Engineer with Django experience")
        print(result)
        # {"is_swe_job": True, "confidence": 0.94, "label": "SWE Job"}
    """

    def __init__(self, model: TextCNN, tokenizer: BPETokenizer, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer: BPETokenizer) -> "ClassifierPredictor":
        """Load predictor from saved checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["model_config"]

        model = TextCNN(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_filters=config["num_filters"],
            filter_sizes=config["filter_sizes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Classifier loaded from {checkpoint_path}")
        return cls(model, tokenizer)

    def predict(self, text: str) -> Dict:
        """
        Predict whether a job posting is a SWE role

        Args:
            text: Raw job description text

        Returns:
            {
                "is_swe_job":  bool,
                "confidence":  float (0.0 - 1.0),
                "label":       "SWE Job" or "Not SWE"
            }
        """
        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        token_ids = token_ids[:self.max_length]
        token_ids += [0] * (self.max_length - len(token_ids))

        # Predict
        tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        confidence = self.model.predict_proba(tensor).item()

        is_swe = confidence >= 0.5

        return {
            "is_swe_job":  is_swe,
            "confidence":  round(confidence, 3),
            "label":       "SWE Job" if is_swe else "Not SWE",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict for multiple job postings"""
        return [self.predict(text) for text in texts]

    def filter_swe_jobs(self, jobs: List[Dict], text_field: str = "description") -> List[Dict]:
        """
        Filter a list of job dicts, keeping only SWE roles

        Args:
            jobs:       List of job dicts with a text field
            text_field: Key in each dict that has the job text

        Returns:
            Filtered list containing only SWE jobs
        """
        swe_jobs = []
        for job in jobs:
            text = job.get(text_field, "")
            result = self.predict(text)
            if result["is_swe_job"]:
                job["classifier_confidence"] = result["confidence"]
                swe_jobs.append(job)
        return swe_jobs
