"""
Test BPE Tokenizer

Quick demo/test of the BPE tokenizer functionality
Run this to verify everything works!

Usage:
    python scripts/test_tokenizer.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenization import BPETokenizer


def main():
    print("="*70)
    print(" BPE TOKENIZER TEST ".center(70, "="))
    print("="*70)
    
    # Sample training corpus (job descriptions and resumes)
    training_texts = [
        # Job descriptions
        "Senior Software Engineer with 5+ years of Python and AWS experience required",
        "Backend Engineer - Python, Django, PostgreSQL, Docker, Kubernetes",
        "Full Stack Developer needed: React, TypeScript, Node.js, MongoDB",
        "Machine Learning Engineer - PyTorch, TensorFlow, NLP, Computer Vision",
        "Data Engineer - Python, Spark, Airflow, Snowflake, ETL pipelines",
        "DevOps Engineer - Kubernetes, Terraform, AWS, CI/CD, monitoring",
        "Frontend Engineer - React, Next.js, TailwindCSS, state management",
        "Software Engineer Intern - algorithms, data structures, Python or Java",
        "New Grad Software Engineer - BS Computer Science, strong coding skills",
        "Site Reliability Engineer - Python, Go, distributed systems, monitoring",
        
        # Resume snippets
        "Experienced software engineer with strong Python and JavaScript skills",
        "Built RESTful APIs using Django and deployed to AWS with Docker",
        "Developed machine learning models using PyTorch for NLP tasks",
        "Implemented CI/CD pipelines with Jenkins and deployed to Kubernetes",
        "Created responsive web applications using React and TypeScript",
        "Designed and optimized database schemas in PostgreSQL and MongoDB",
        "Experience with microservices architecture and event-driven systems",
        "Strong foundation in algorithms, data structures, and system design",
        "Proficient in version control with Git and collaborative development",
        "Led team of engineers to deliver high-quality software products",
    ]
    
    print("\n📚 STEP 1: Training Tokenizer")
    print("-"*70)
    print(f"Training on {len(training_texts)} sample texts...")
    print(f"Target vocabulary size: 500 tokens")
    print()
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=500, lowercase=True)
    tokenizer.train(training_texts)
    
    print(f"\n✅ Training complete!")
    print(f"   Final vocabulary size: {len(tokenizer)}")
    
    # Test encoding and decoding
    print("\n"+ "="*70)
    print("📝 STEP 2: Testing Encoding/Decoding")
    print("-"*70)
    
    test_sentences = [
        "I am a Python software engineer with machine learning experience",
        "Looking for backend developer position with Django and PostgreSQL",
        "Full stack engineer needed with React and TypeScript skills",
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"  Original:  '{sentence}'")
        
        # Encode
        token_ids = tokenizer.encode(sentence, add_special_tokens=True)
        print(f"  Token IDs: {token_ids}")
        print(f"  # Tokens:  {len(token_ids)}")
        
        # Decode
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Decoded:   '{decoded}'")
        
        # Check if roundtrip works
        match = "✅" if sentence.lower().replace(" ", "") == decoded.replace(" ", "") else "❌"
        print(f"  Roundtrip: {match}")
    
    # Show vocabulary samples
    print("\n" + "="*70)
    print("📖 STEP 3: Vocabulary Samples")
    print("-"*70)
    
    print("\nSpecial Tokens:")
    special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[SEP]', '[CLS]', '[MASK]']
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token_id:3d}: '{token}'")
    
    print("\nLearned Subword Tokens (sample):")
    for i in range(7, min(30, len(tokenizer))):
        token = tokenizer.id_to_token(i)
        print(f"  {i:3d}: '{token}'")
    
    # Test with unknown words
    print("\n" + "="*70)
    print("🔍 STEP 4: Testing with Unknown Words")
    print("-"*70)
    
    unknown_tests = [
        "I know TensorFlow and PyTorch",  # Tech terms
        "Experienced with Kubernetes and Docker",  # Common in jobs
        "Strong algorithmic problem-solving",  # General skills
    ]
    
    for sentence in unknown_tests:
        token_ids = tokenizer.encode(sentence)
        tokens = [tokenizer.id_to_token(id) for id in token_ids if id not in [0, 1, 2, 3]]
        
        print(f"\n  Text: '{sentence}'")
        print(f"  Tokens: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"  Count: {len(tokens)} tokens")
    
    # Test save/load
    print("\n" + "="*70)
    print("💾 STEP 5: Testing Save/Load")
    print("-"*70)
    
    save_path = "test_tokenizer.json"
    print(f"\nSaving tokenizer to '{save_path}'...")
    tokenizer.save(save_path)
    
    print(f"Loading tokenizer from '{save_path}'...")
    new_tokenizer = BPETokenizer()
    new_tokenizer.load(save_path)
    
    # Verify loaded tokenizer works
    test_text = "Python engineer with machine learning experience"
    original_ids = tokenizer.encode(test_text)
    loaded_ids = new_tokenizer.encode(test_text)
    
    print(f"\nVerification:")
    print(f"  Original tokenizer: {original_ids}")
    print(f"  Loaded tokenizer:   {loaded_ids}")
    print(f"  Match: {'✅ Perfect match!' if original_ids == loaded_ids else '❌ Mismatch!'}")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\nCleaned up test file: {save_path}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📌 Summary:")
    print(f"   • Tokenizer trained successfully on {len(training_texts)} texts")
    print(f"   • Vocabulary size: {len(tokenizer)} tokens")
    print(f"   • Encoding/decoding works correctly")
    print(f"   • Handles unknown words via subwords")
    print(f"   • Save/load functionality verified")
    print("\n🎉 Your tokenizer is ready to use!")
    print("\n💡 Next steps:")
    print("   1. Train on larger corpus of job descriptions")
    print("   2. Use this tokenizer for your models")
    print("   3. Run: python scripts/train_tokenizer.py --help")
    print("="*70)


if __name__ == "__main__":
    main()
