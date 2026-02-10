"""
NER Inference - Extract Entities from Job Postings

Given a raw job description, returns structured entities:
    {
        "technologies": ["Python", "Docker", "AWS"],
        "skills":       ["problem solving", "communication"],
        "requirements": ["5+ years experience", "Bachelor's degree"],
        "levels":       ["Senior"]
    }
"""
import torch
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.ner_model import BiLSTMCRF
from src.data.ner_dataset import ID2LABEL, NUM_LABELS
from src.tokenization import BPETokenizer


class NERPredictor:
    """
    Runs NER inference on raw job description text

    Usage:
        predictor = NERPredictor.from_checkpoint("models/ner/checkpoints/best_model.pt", tokenizer)
        results = predictor.extract("We need a Senior Python Engineer with Docker experience")
        print(results)
        # {
        #   "technologies": ["Python", "Docker"],
        #   "skills": [],
        #   "requirements": [],
        #   "levels": ["Senior"]
        # }
    """

    def __init__(self, model: BiLSTMCRF, tokenizer: BPETokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Use MPS if available (M1 GPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer: BPETokenizer) -> "NERPredictor":
        """Load predictor from saved checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["model_config"]

        model = BiLSTMCRF(
            vocab_size=config["vocab_size"],
            num_labels=config["num_labels"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Model loaded from {checkpoint_path}")
        return cls(model, tokenizer)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from job description text

        Args:
            text: Raw job description string

        Returns:
            Dict with entity lists by type
        """
        # Tokenize
        words = text.split()
        token_ids = []
        word_to_token_start = []   # maps word index → first token index

        for word in words:
            word_to_token_start.append(len(token_ids))
            ids = self.tokenizer.encode(word.lower(), add_special_tokens=False)
            token_ids.extend(ids if ids else [self.tokenizer.token_to_id("[UNK]")])

        if not token_ids:
            return {"technologies": [], "skills": [], "requirements": [], "levels": []}

        # Prepare tensors
        token_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        mask_tensor  = torch.ones(1, len(token_ids), dtype=torch.float).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model.predict(token_tensor, mask_tensor)[0]

        # Map token predictions back to words
        word_labels = []
        for word_idx in range(len(words)):
            token_start = word_to_token_start[word_idx]
            if token_start < len(predictions):
                word_labels.append(predictions[token_start])
            else:
                word_labels.append(0)  # O

        # Extract entity spans
        entities = self._extract_spans(words, word_labels)
        return entities

    def extract_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """Extract entities from multiple texts"""
        return [self.extract(text) for text in texts]

    def _extract_spans(
        self,
        words: List[str],
        label_ids: List[int],
    ) -> Dict[str, List[str]]:
        """
        Convert BIO label sequence to entity spans

        Args:
            words:     Original word tokens
            label_ids: Predicted label IDs per word

        Returns:
            Dict grouping entities by type
        """
        entities = {
            "technologies": [],
            "skills":       [],
            "requirements": [],
            "levels":       [],
        }

        type_to_key = {
            "TECH":        "technologies",
            "SKILL":       "skills",
            "REQUIREMENT": "requirements",
            "LEVEL":       "levels",
        }

        current_entity_words = []
        current_entity_type = None

        for word, label_id in zip(words, label_ids):
            label = ID2LABEL.get(label_id, "O")

            if label.startswith("B-"):
                # Save previous entity if any
                if current_entity_words and current_entity_type:
                    entity_text = " ".join(current_entity_words)
                    key = type_to_key.get(current_entity_type)
                    if key and entity_text not in entities[key]:
                        entities[key].append(entity_text)

                # Start new entity
                current_entity_type = label[2:]
                current_entity_words = [word]

            elif label.startswith("I-") and current_entity_type:
                # Continue entity (only if types match)
                if label[2:] == current_entity_type:
                    current_entity_words.append(word)
                else:
                    # Type mismatch - save current and reset
                    if current_entity_words:
                        entity_text = " ".join(current_entity_words)
                        key = type_to_key.get(current_entity_type)
                        if key and entity_text not in entities[key]:
                            entities[key].append(entity_text)
                    current_entity_words = []
                    current_entity_type = None

            else:
                # O label - save current entity and reset
                if current_entity_words and current_entity_type:
                    entity_text = " ".join(current_entity_words)
                    key = type_to_key.get(current_entity_type)
                    if key and entity_text not in entities[key]:
                        entities[key].append(entity_text)
                current_entity_words = []
                current_entity_type = None

        # Don't forget last entity
        if current_entity_words and current_entity_type:
            entity_text = " ".join(current_entity_words)
            key = type_to_key.get(current_entity_type)
            if key and entity_text not in entities[key]:
                entities[key].append(entity_text)

        return entities
