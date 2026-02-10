"""
Base Tokenizer Class

All tokenizers inherit from this class
Defines the common interface for tokenization
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
from pathlib import Path


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers
    
    Defines the interface that all tokenizers must implement
    """
    
    # Special tokens used in all tokenizers
    PAD_TOKEN = "[PAD]"      # Padding token
    UNK_TOKEN = "[UNK]"      # Unknown token
    BOS_TOKEN = "[BOS]"      # Beginning of sequence
    EOS_TOKEN = "[EOS]"      # End of sequence
    SEP_TOKEN = "[SEP]"      # Separator token
    CLS_TOKEN = "[CLS]"      # Classification token
    MASK_TOKEN = "[MASK]"    # Mask token (for masked language modeling)
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize tokenizer
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Token to ID mapping
        self.token2id: Dict[str, int] = {}
        
        # ID to token mapping
        self.id2token: Dict[int, str] = {}
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.SEP_TOKEN,
            self.CLS_TOKEN,
            self.MASK_TOKEN,
        ]
        
        for i, token in enumerate(special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token
    
    @abstractmethod
    def train(self, texts: List[str]):
        """
        Train tokenizer on a corpus of texts
        
        Args:
            texts: List of text strings to train on
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """
        Decode multiple sequences of token IDs
        
        Args:
            token_ids_batch: List of token ID lists
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(token_ids) for token_ids in token_ids_batch]
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.token2id)
    
    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its ID
        
        Args:
            token: Token string
            
        Returns:
            Token ID (or UNK token ID if not in vocab)
        """
        return self.token2id.get(token, self.token2id[self.UNK_TOKEN])
    
    def id_to_token(self, token_id: int) -> str:
        """
        Convert a token ID to its token
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string (or UNK token if ID not in vocab)
        """
        return self.id2token.get(token_id, self.UNK_TOKEN)
    
    def save(self, filepath: str):
        """
        Save tokenizer to file
        
        Args:
            filepath: Path to save tokenizer
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary and configuration
        save_data = {
            'vocab_size': self.vocab_size,
            'token2id': self.token2id,
            'id2token': {int(k): v for k, v in self.id2token.items()},  # JSON keys must be strings
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load tokenizer from file
        
        Args:
            filepath: Path to load tokenizer from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        self.vocab_size = save_data['vocab_size']
        self.token2id = save_data['token2id']
        self.id2token = {int(k): v for k, v in save_data['id2token'].items()}
        
        print(f"Tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {self.get_vocab_size()}")
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.get_vocab_size()
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(vocab_size={self.get_vocab_size()})"
