"""
NER Labels and Dataset - Built from Scratch

Label Scheme (BIO tagging):
    B-  = Beginning of an entity
    I-  = Inside (continuation) of an entity
    O   = Outside (not an entity)

Example:
    "Python  and  Docker  required  for  Senior  engineers"
     B-TECH   O   B-TECH     O       O   B-LEVEL     O

Entity Types:
    TECH         Technologies/tools   (Python, Docker, AWS, React)
    SKILL        Soft/hard skills     (problem solving, leadership)
    REQUIREMENT  Experience/degree    (5+ years, Bachelor's degree)
    LEVEL        Seniority            (Senior, Junior, Lead, Staff)
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.tokenization import BPETokenizer


# ── Label Definitions ─────────────────────────────────────────────────────────

LABELS = [
    "O",             # 0 - Outside
    "B-TECH",        # 1 - Beginning of Technology
    "I-TECH",        # 2 - Inside Technology
    "B-SKILL",       # 3 - Beginning of Skill
    "I-SKILL",       # 4 - Inside Skill
    "B-REQUIREMENT", # 5 - Beginning of Requirement
    "I-REQUIREMENT", # 6 - Inside Requirement
    "B-LEVEL",       # 7 - Beginning of Level
    "I-LEVEL",       # 8 - Inside Level
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


# ── Sample Training Data ───────────────────────────────────────────────────────
# Format: list of (words, labels) tuples
# This is what annotated data looks like
# In production you'd annotate thousands of job postings

SAMPLE_DATA = [
    (
        ["We", "are", "looking", "for", "a", "Senior", "Python", "Engineer"],
        ["O",  "O",   "O",       "O",   "O", "B-LEVEL","B-TECH", "O"]
    ),
    (
        ["Requirements", ":", "5+", "years", "of", "Python", "and", "Django", "experience"],
        ["O",             "O", "B-REQUIREMENT", "I-REQUIREMENT", "O", "B-TECH", "O", "B-TECH", "O"]
    ),
    (
        ["Strong", "problem", "solving", "and", "communication", "skills", "required"],
        ["O",      "B-SKILL", "I-SKILL", "O",   "B-SKILL",       "O",      "O"]
    ),
    (
        ["Experience", "with", "Docker", ",", "Kubernetes", "and", "AWS"],
        ["O",          "O",    "B-TECH", "O", "B-TECH",     "O",   "B-TECH"]
    ),
    (
        ["Bachelor", "'s", "degree", "in", "Computer", "Science", "required"],
        ["B-REQUIREMENT", "I-REQUIREMENT", "I-REQUIREMENT", "O", "O", "O", "O"]
    ),
    (
        ["Lead", "Engineer", "position", "with", "React", "and", "TypeScript"],
        ["B-LEVEL", "O",     "O",        "O",    "B-TECH", "O",  "B-TECH"]
    ),
    (
        ["Looking", "for", "Junior", "developer", "with", "Python", "skills"],
        ["O",       "O",   "B-LEVEL","O",         "O",    "B-TECH", "O"]
    ),
    (
        ["Must", "have", "3+", "years", "experience", "with", "PostgreSQL", "and", "Redis"],
        ["O",    "O",    "B-REQUIREMENT", "I-REQUIREMENT", "O", "O", "B-TECH", "O", "B-TECH"]
    ),
    (
        ["Strong", "leadership", "and", "teamwork", "abilities", "needed"],
        ["O",      "B-SKILL",    "O",   "B-SKILL",  "O",         "O"]
    ),
    (
        ["Staff", "engineer", "role", "requiring", "Go", "and", "Rust", "expertise"],
        ["B-LEVEL","O",       "O",    "O",         "B-TECH","O", "B-TECH", "O"]
    ),
    (
        ["Proficiency", "in", "PyTorch", "or", "TensorFlow", "for", "ML", "work"],
        ["O",           "O",  "B-TECH",  "O",  "B-TECH",     "O",   "B-TECH", "O"]
    ),
    (
        ["2+", "years", "of", "cloud", "experience", "AWS", "GCP", "or", "Azure"],
        ["B-REQUIREMENT","I-REQUIREMENT","O","O","O","B-TECH","B-TECH","O","B-TECH"]
    ),
    (
        ["Excellent", "written", "and", "verbal", "communication", "skills"],
        ["O",         "B-SKILL", "O",   "B-SKILL","I-SKILL",       "O"]
    ),
    (
        ["Principal", "engineer", "with", "10+", "years", "of", "experience"],
        ["B-LEVEL",   "O",        "O",    "B-REQUIREMENT","I-REQUIREMENT","O","O"]
    ),
    (
        ["Solid", "understanding", "of", "algorithms", "and", "data", "structures"],
        ["O",     "O",             "O",  "B-SKILL",    "O",   "B-SKILL","I-SKILL"]
    ),
]


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER training

    Converts (words, labels) examples into token IDs and label IDs
    """

    def __init__(
        self,
        examples: List[Tuple[List[str], List[str]]],
        tokenizer: BPETokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            examples:   List of (words, labels) tuples
            tokenizer:  Trained BPE tokenizer
            max_length: Maximum sequence length (truncate longer)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._process(examples)

    def _process(
        self,
        examples: List[Tuple[List[str], List[str]]]
    ) -> List[Dict]:
        """
        Convert words+labels → token_ids, label_ids, mask

        Note: BPE may split one word into multiple tokens.
        We assign the word's label to the FIRST subword token,
        and use label O for remaining subword tokens.
        """
        processed = []

        for words, word_labels in examples:
            token_ids = []
            label_ids = []

            for word, label in zip(words, word_labels):
                # Tokenize the word (may produce multiple subword tokens)
                word_token_ids = self.tokenizer.encode(
                    word, add_special_tokens=False
                )

                if not word_token_ids:
                    continue

                # First subword gets the real label
                token_ids.append(word_token_ids[0])
                label_ids.append(LABEL2ID[label])

                # Remaining subwords get O (they're continuations, not new entities)
                for extra_token in word_token_ids[1:]:
                    token_ids.append(extra_token)
                    label_ids.append(LABEL2ID["O"])

            # Truncate to max_length
            token_ids = token_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]

            # Create attention mask (1 for real tokens)
            mask = [1] * len(token_ids)

            processed.append({
                "token_ids": token_ids,
                "label_ids": label_ids,
                "mask": mask,
            })

        return processed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad sequences in a batch to the same length

    Called automatically by DataLoader to form batches
    """
    max_len = max(len(item["token_ids"]) for item in batch)

    token_ids_batch = []
    label_ids_batch = []
    mask_batch = []

    for item in batch:
        length = len(item["token_ids"])
        pad_len = max_len - length

        # Pad with zeros ([PAD] token = index 0)
        token_ids_batch.append(item["token_ids"] + [0] * pad_len)
        label_ids_batch.append(item["label_ids"] + [0] * pad_len)
        mask_batch.append(item["mask"] + [0] * pad_len)

    return {
        "token_ids": torch.tensor(token_ids_batch, dtype=torch.long),
        "label_ids": torch.tensor(label_ids_batch, dtype=torch.long),
        "mask":      torch.tensor(mask_batch,      dtype=torch.float),
    }
