"""
Train BPE Tokenizer on Job Description Data

This script trains a BPE tokenizer from scratch on your job descriptions
and resumes. Run this before training any models.

Usage:
    python scripts/train_tokenizer.py --data_dir data/raw/jobs --vocab_size 10000
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenization import BPETokenizer
import argparse
import json


def load_job_data(data_dir: str) -> list:
    """
    Load job descriptions from directory
    
    Args:
        data_dir: Directory containing job JSON files
        
    Returns:
        List of job description texts
    """
    data_path = Path(data_dir)
    texts = []
    
    if not data_path.exists():
        print(f"Warning: {data_dir} does not exist yet")
        print("Using sample data for demonstration...")
        
        # Sample job descriptions for testing
        texts = [
            "We are looking for a Senior Software Engineer with 5+ years of experience in Python, Django, and AWS. "
            "The ideal candidate should have strong knowledge of RESTful APIs, microservices architecture, and Docker.",
            
            "Backend Engineer position available. Requirements: Python, FastAPI, PostgreSQL, Redis. "
            "Experience with Kubernetes and CI/CD pipelines preferred. Strong problem-solving skills required.",
            
            "Full Stack Developer needed. Must have: React, TypeScript, Node.js, Python. "
            "Experience with MongoDB, GraphQL, and cloud platforms (AWS/GCP) is a plus.",
            
            "Machine Learning Engineer role. Required skills: Python, PyTorch, TensorFlow, scikit-learn. "
            "Experience with NLP, computer vision, and deploying ML models in production.",
            
            "Data Engineer position. Tech stack: Python, Apache Spark, Airflow, Snowflake. "
            "Experience with data pipelines, ETL processes, and big data technologies required.",
            
            "DevOps Engineer wanted. Skills needed: Kubernetes, Docker, Terraform, AWS. "
            "Experience with CI/CD, monitoring tools (Prometheus, Grafana), and infrastructure as code.",
            
            "Frontend Engineer position. Required: React, TypeScript, Next.js, TailwindCSS. "
            "Experience with state management (Redux, Zustand) and testing (Jest, Cypress).",
            
            "Site Reliability Engineer role. Need: Python, Go, Kubernetes, monitoring tools. "
            "Experience with distributed systems, high availability, and incident management.",
            
            "Software Engineer Intern - Summer 2025. Looking for students with coursework in: "
            "algorithms, data structures, object-oriented programming. Python or Java preferred.",
            
            "New Grad Software Engineer. Requirements: BS in Computer Science or equivalent. "
            "Strong coding skills in Python, Java, or C++. Knowledge of algorithms and system design.",
        ]
        
        return texts
    
    # Load from JSON files
    for filepath in data_path.glob('**/*.json'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract job description
                if isinstance(data, dict):
                    if 'description' in data:
                        texts.append(data['description'])
                    elif 'text' in data:
                        texts.append(data['text'])
                    elif 'content' in data:
                        texts.append(data['content'])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'description' in item:
                            texts.append(item['description'])
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return texts


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw/jobs',
        help='Directory containing job description data'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=10000,
        help='Target vocabulary size'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/vocabularies/bpe_tokenizer.json',
        help='Output path for trained tokenizer'
    )
    parser.add_argument(
        '--lowercase',
        action='store_true',
        default=True,
        help='Convert text to lowercase'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BPE Tokenizer Training")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Lowercase: {args.lowercase}")
    print(f"Output: {args.output}")
    print("="*60)
    print()
    
    # Load training data
    print("Loading training data...")
    texts = load_job_data(args.data_dir)
    print(f"Loaded {len(texts)} job descriptions")
    
    if len(texts) == 0:
        print("Error: No training data found!")
        return
    
    # Show sample
    print(f"\nSample text (first 200 chars):")
    print(f"  {texts[0][:200]}...")
    print()
    
    # Create and train tokenizer
    print("Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size, lowercase=args.lowercase)
    
    print("\nTraining tokenizer (this may take a few minutes)...")
    tokenizer.train(texts)
    
    # Save tokenizer
    print(f"\nSaving tokenizer to {args.output}...")
    tokenizer.save(args.output)
    
    # Test the tokenizer
    print("\n" + "="*60)
    print("Testing Tokenizer")
    print("="*60)
    
    test_sentences = [
        "I am a Python software engineer",
        "Looking for backend developer with Django experience",
        "Machine learning engineer position available",
    ]
    
    for sentence in test_sentences:
        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)
        
        print(f"\nOriginal:  {sentence}")
        print(f"Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        print(f"Decoded:   {decoded}")
        print(f"Tokens:    {len(token_ids)}")
    
    # Show vocabulary statistics
    print("\n" + "="*60)
    print("Vocabulary Statistics")
    print("="*60)
    print(f"Total vocabulary size: {len(tokenizer)}")
    print(f"Number of learned merges: {len(tokenizer.merges)}")
    
    print(f"\nFirst 30 tokens in vocabulary:")
    for i in range(min(30, len(tokenizer))):
        token = tokenizer.id_to_token(i)
        print(f"  {i:4d}: '{token}'")
    
    print("\n" + "="*60)
    print("Tokenizer training complete!")
    print(f"Saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
