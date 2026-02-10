"""
Byte-Pair Encoding (BPE) Tokenizer - Built from Scratch

BPE is a subword tokenization algorithm that learns to merge 
frequently occurring character pairs.

Algorithm:
1. Start with characters as initial vocabulary
2. Find most frequent pair of tokens
3. Merge that pair into a new token
4. Repeat until vocab_size is reached

This handles unknown words by breaking them into subwords.
"""
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re
from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding Tokenizer
    
    Learns subword vocabulary from training corpus
    """
    
    def __init__(self, vocab_size: int = 10000, lowercase: bool = True):
        """
        Initialize BPE tokenizer
        
        Args:
            vocab_size: Target vocabulary size
            lowercase: Whether to lowercase text
        """
        super().__init__(vocab_size)
        
        self.lowercase = lowercase
        
        # BPE merge rules: (token1, token2) -> merged_token
        self.merges: Dict[Tuple[str, str], str] = {}
        
        # Learned vocabulary (set of all tokens)
        self.vocab: Set[str] = set()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Add spaces around punctuation for better tokenization
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _get_words(self, texts: List[str]) -> Dict[str, int]:
        """
        Split texts into words and count frequencies
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary of word -> frequency
        """
        word_freqs = Counter()
        
        for text in texts:
            # Preprocess
            text = self._preprocess_text(text)
            
            # Split into words
            words = text.split()
            
            # Count word frequencies
            word_freqs.update(words)
        
        return dict(word_freqs)
    
    def _get_char_vocab(self, word_freqs: Dict[str, int]) -> Set[str]:
        """
        Get initial character-level vocabulary
        
        Args:
            word_freqs: Word frequency dictionary
            
        Returns:
            Set of all characters in the corpus
        """
        char_vocab = set()
        
        for word in word_freqs.keys():
            # Add each character
            char_vocab.update(list(word))
        
        return char_vocab
    
    def _get_pair_frequencies(
        self,
        word_freqs: Dict[str, int],
        splits: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count frequencies of all adjacent token pairs
        
        Args:
            word_freqs: Word frequencies
            splits: Current token splits for each word
            
        Returns:
            Dictionary of (token1, token2) -> frequency
        """
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            
            # Count pairs
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return dict(pair_freqs)
    
    def _merge_pair(
        self,
        pair: Tuple[str, str],
        splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Merge a token pair in all word splits
        
        Args:
            pair: Pair of tokens to merge
            splits: Current token splits
            
        Returns:
            Updated splits with merged pair
        """
        new_splits = {}
        
        for word, split in splits.items():
            new_split = []
            i = 0
            
            while i < len(split):
                # Check if current and next token match the pair
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    # Merge the pair
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            new_splits[word] = new_split
        
        return new_splits
    
    def train(self, texts: List[str]):
        """
        Train BPE tokenizer on corpus
        
        Args:
            texts: List of training texts
        """
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        
        # Step 1: Get word frequencies
        print("1. Counting word frequencies...")
        word_freqs = self._get_words(texts)
        print(f"   Found {len(word_freqs)} unique words")
        
        # Step 2: Initialize with character-level splits
        print("2. Initializing character vocabulary...")
        char_vocab = self._get_char_vocab(word_freqs)
        print(f"   Character vocabulary size: {len(char_vocab)}")
        
        # Initialize splits: each word split into characters
        splits = {word: list(word) for word in word_freqs.keys()}
        
        # Add characters to vocabulary
        self.vocab.update(char_vocab)
        
        # Step 3: Learn merges
        print(f"3. Learning merges (target vocab size: {self.vocab_size})...")
        
        num_merges_needed = self.vocab_size - len(self.vocab) - len(self.token2id)
        
        for i in range(num_merges_needed):
            # Get pair frequencies
            pair_freqs = self._get_pair_frequencies(word_freqs, splits)
            
            if not pair_freqs:
                print(f"   No more pairs to merge at iteration {i}")
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Merge the pair
            merged_token = best_pair[0] + best_pair[1]
            
            # Store merge rule
            self.merges[best_pair] = merged_token
            
            # Add to vocabulary
            self.vocab.add(merged_token)
            
            # Update splits
            splits = self._merge_pair(best_pair, splits)
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"   Learned {i + 1} merges, vocab size: {len(self.vocab) + len(self.token2id)}")
        
        # Step 4: Build token2id and id2token mappings
        print("4. Building vocabulary mappings...")
        
        # Special tokens are already in token2id from base class
        current_id = len(self.token2id)
        
        # Add all vocabulary tokens
        for token in sorted(self.vocab):
            if token not in self.token2id:
                self.token2id[token] = current_id
                self.id2token[current_id] = token
                current_id += 1
        
        print(f"\nTraining complete!")
        print(f"Final vocabulary size: {len(self.token2id)}")
        print(f"Number of merges learned: {len(self.merges)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of subword tokens
        """
        # Start with character split
        tokens = list(word)
        
        # Apply merges in order they were learned
        for pair, merged in self.merges.items():
            i = 0
            new_tokens = []
            
            while i < len(tokens):
                # Check if current and next match the pair
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    # Merge
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Preprocess
        text = self._preprocess_text(text)
        
        # Split into words
        words = text.split()
        
        # Tokenize each word
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        # Convert tokens to IDs
        token_ids = [self.token_to_id(token) for token in tokens]
        
        # Add special tokens if requested
        if add_special_tokens:
            token_ids = [self.token2id[self.BOS_TOKEN]] + token_ids + [self.token2id[self.EOS_TOKEN]]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # Convert IDs to tokens
        tokens = [self.id_to_token(id) for id in token_ids]
        
        # Remove special tokens if requested
        if skip_special_tokens:
            special_tokens = {
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.SEP_TOKEN,
                self.CLS_TOKEN,
                self.MASK_TOKEN,
            }
            tokens = [t for t in tokens if t not in special_tokens]
        
        # Join tokens
        text = ''.join(tokens)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text
    
    def save(self, filepath: str):
        """
        Save tokenizer to file
        
        Args:
            filepath: Path to save
        """
        import json
        from pathlib import Path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert merges dict for JSON serialization
        merges_list = [
            {
                'pair': list(pair),
                'merged': merged
            }
            for pair, merged in self.merges.items()
        ]
        
        save_data = {
            'vocab_size': self.vocab_size,
            'lowercase': self.lowercase,
            'token2id': self.token2id,
            'id2token': {int(k): v for k, v in self.id2token.items()},
            'merges': merges_list,
            'vocab': list(self.vocab),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"BPE tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load tokenizer from file
        
        Args:
            filepath: Path to load from
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        self.vocab_size = save_data['vocab_size']
        self.lowercase = save_data['lowercase']
        self.token2id = save_data['token2id']
        self.id2token = {int(k): v for k, v in save_data['id2token'].items()}
        
        # Reconstruct merges
        self.merges = {
            tuple(item['pair']): item['merged']
            for item in save_data['merges']
        }
        
        self.vocab = set(save_data['vocab'])
        
        print(f"BPE tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {len(self.token2id)}")
        print(f"Number of merges: {len(self.merges)}")


# Example usage
if __name__ == "__main__":
    # Sample training data
    texts = [
        "I am a software engineer with experience in Python and machine learning",
        "Looking for backend engineer position with Python Django experience",
        "Senior machine learning engineer role requires deep learning expertise",
        "Full stack engineer needed with React and Python skills",
        "Data engineer position available for experienced Python developers",
    ]
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=500, lowercase=True)
    tokenizer.train(texts)
    
    # Test encoding
    test_text = "I am a Python engineer"
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"\nOriginal: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")
    
    # Show some tokens
    print(f"\nFirst 20 tokens in vocabulary:")
    for i in range(20):
        print(f"  {i}: '{tokenizer.id_to_token(i)}'")
