"""
Job Classifier Dataset - Built from Scratch

Label Scheme (binary):
    1 = Software Engineering job  (apply to this)
    0 = Not a SWE job             (skip this)

The model learns to distinguish SWE jobs from:
    - Marketing, Sales, HR, Finance roles
    - Non-tech jobs (designer, writer, manager)
    - Adjacent tech (IT support, QA manual tester)
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.tokenization import BPETokenizer


# ── Training Data ──────────────────────────────────────────────────────────────
# Format: (job_text, label)
# 1 = SWE job, 0 = not SWE
# In production: use your scraped jobs + manual labels

SAMPLE_DATA: List[Tuple[str, int]] = [

    # ── Positive examples (SWE jobs = 1) ──────────────────────────────────
    ("Senior Software Engineer Python Django REST API microservices AWS Docker Kubernetes", 1),
    ("Backend Engineer position FastAPI PostgreSQL Redis celery distributed systems", 1),
    ("Full Stack Developer React TypeScript Node.js MongoDB GraphQL", 1),
    ("Machine Learning Engineer PyTorch TensorFlow NLP computer vision model deployment", 1),
    ("Data Engineer Apache Spark Airflow Snowflake ETL pipeline Python SQL", 1),
    ("DevOps Engineer Kubernetes Terraform AWS CI CD Jenkins infrastructure as code", 1),
    ("Frontend Engineer React Next.js TypeScript TailwindCSS state management", 1),
    ("Site Reliability Engineer Python Go distributed systems monitoring Prometheus", 1),
    ("Staff Engineer technical leadership architecture distributed systems Python", 1),
    ("Principal Software Engineer system design large scale microservices Java Python", 1),
    ("Android Engineer Kotlin Java mobile development Android SDK", 1),
    ("iOS Engineer Swift Objective C mobile Apple developer", 1),
    ("Security Engineer penetration testing vulnerability cloud security Python", 1),
    ("Embedded Systems Engineer C C++ firmware RTOS real time systems", 1),
    ("Compiler Engineer LLVM systems programming C++ optimization", 1),
    ("Platform Engineer developer tooling infrastructure Python Golang Kubernetes", 1),
    ("New Grad Software Engineer algorithms data structures Python Java computer science", 1),
    ("Software Engineer Intern Python Java algorithms coding computer science student", 1),
    ("Senior Backend Engineer Rust Go performance distributed systems databases", 1),
    ("Cloud Engineer AWS GCP Azure Terraform infrastructure automation Python", 1),
    ("Database Engineer PostgreSQL MySQL performance optimization query tuning", 1),
    ("API Engineer REST GraphQL microservices documentation Python FastAPI", 1),
    ("Game Engineer Unity C# Unreal Engine graphics rendering performance", 1),
    ("Blockchain Engineer Solidity Ethereum smart contracts Web3 Python", 1),
    ("Computer Vision Engineer OpenCV PyTorch deep learning image processing", 1),

    # ── Negative examples (NOT SWE jobs = 0) ──────────────────────────────
    ("Marketing Manager digital marketing campaigns social media brand strategy SEO", 0),
    ("Sales Representative B2B enterprise sales CRM Salesforce quota revenue", 0),
    ("Product Manager roadmap stakeholders user research agile sprint planning", 0),
    ("UX Designer Figma user research wireframes prototyping usability testing", 0),
    ("Graphic Designer Adobe Illustrator Photoshop brand identity visual design", 0),
    ("HR Manager recruiting onboarding employee relations benefits compensation", 0),
    ("Finance Analyst Excel financial modeling forecasting budget variance", 0),
    ("Content Writer SEO blog articles copywriting editorial calendar", 0),
    ("Customer Success Manager onboarding retention NPS customer health", 0),
    ("IT Support Helpdesk Windows Active Directory ticketing hardware troubleshooting", 0),
    ("Project Manager PMP agile waterfall stakeholder budget timeline delivery", 0),
    ("Data Analyst Excel Tableau Power BI SQL reporting dashboards business intelligence", 0),
    ("Operations Manager supply chain logistics process improvement lean six sigma", 0),
    ("Business Analyst requirements gathering process mapping stakeholder UAT", 0),
    ("Technical Recruiter sourcing screening interviews offer negotiation", 0),
    ("Account Executive enterprise sales pipeline demos closing revenue quota", 0),
    ("Social Media Manager Instagram TikTok content creation engagement community", 0),
    ("QA Tester manual testing test cases bug reports regression Jira", 0),
    ("Network Administrator Cisco switches routers firewall VPN network monitoring", 0),
    ("Office Manager facilities vendor management administrative scheduling", 0),
    ("Legal Counsel contracts compliance intellectual property negotiations", 0),
    ("CFO financial planning investor relations fundraising financial statements", 0),
    ("Physical Therapist patient care rehabilitation exercise prescription clinic", 0),
    ("Registered Nurse patient care hospital clinical documentation EMR", 0),
    ("Teacher curriculum lesson plans classroom management student assessment", 0),
]


class ClassifierDataset(Dataset):
    """
    PyTorch Dataset for job classification training

    Converts (job_text, label) pairs into token IDs and labels
    """

    def __init__(
        self,
        examples: List[Tuple[str, int]],
        tokenizer: BPETokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            examples:   List of (job_text, label) tuples
            tokenizer:  Trained BPE tokenizer
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.examples = self._process(examples, tokenizer)

    def _process(
        self,
        examples: List[Tuple[str, int]],
        tokenizer: BPETokenizer
    ) -> List[Dict]:
        """Tokenize and encode all examples"""
        processed = []

        for text, label in examples:
            # Encode text to token IDs
            token_ids = tokenizer.encode(text, add_special_tokens=True)

            # Truncate if too long
            token_ids = token_ids[:self.max_length]

            # Pad if too short
            pad_len = self.max_length - len(token_ids)
            token_ids = token_ids + [0] * pad_len

            processed.append({
                "token_ids": token_ids,
                "label": float(label),
            })

        return processed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.examples[idx]
        return {
            "token_ids": torch.tensor(item["token_ids"], dtype=torch.long),
            "label":     torch.tensor(item["label"],     dtype=torch.float),
        }


def get_class_weights(examples: List[Tuple[str, int]]) -> torch.Tensor:
    """
    Compute class weights to handle class imbalance

    If you have more non-SWE jobs than SWE jobs in your data,
    this makes the model pay more attention to the minority class.
    """
    labels = [label for _, label in examples]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    # Weight = total / (2 * class_count)
    weight_pos = len(labels) / (2 * num_pos) if num_pos > 0 else 1.0
    weight_neg = len(labels) / (2 * num_neg) if num_neg > 0 else 1.0

    return torch.tensor([weight_neg, weight_pos])
