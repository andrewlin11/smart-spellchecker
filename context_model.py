import math
import re
from collections import defaultdict
from typing import List, Tuple


class NGramLanguageModel:
    """N-gram language model with add-k smoothing for spell checking."""
    
    def __init__(self, n: int = 3, k: float = 0.01):
        self.n = n  # n for n-gram
        self.k = k  # smoothing parameter
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase word tokens for LM consistency."""
        # Lowercase and split on word boundaries; ignore numbers/punct
        # Keep words only to match spellchecker tokens
        return re.findall(r"[a-z]+", text.lower())
    
    def train(self, text: str):
        """Train the n-gram model on the given text."""
        tokens = self.tokenize(text)
        
        # Add start and end tokens
        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        
        # Update vocabulary
        self.vocab.update(padded_tokens)
        
        # Count n-grams and contexts
        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i + self.n])
            context = ngram[:-1]
            
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1
        
        self.vocab_size = len(self.vocab)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get probability of an n-gram with add-k smoothing."""
        context = ngram[:-1]
        
        # Add-k smoothing formula:
        # P(w|context) = (count(context, w) + k) / (count(context) + k * V)
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        
        # Handle unseen contexts
        if context_count == 0:
            return self.k / (self.k * self.vocab_size)
        
        probability = (ngram_count + self.k) / (context_count + self.k * self.vocab_size)
        return probability
    
    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of the given text under this model."""
        tokens = self.tokenize(text)
        if len(tokens) == 0:
            return float('inf')

        padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

        log_prob_sum = 0.0
        num_tokens = 0

        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i + self.n])
            prob = self.get_probability(ngram)
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                log_prob_sum += math.log(1e-10)
            num_tokens += 1

        if num_tokens == 0:
            return float('inf')

        avg_log_prob = log_prob_sum / num_tokens
        perplexity = math.pow(2, -avg_log_prob)
        return perplexity
    
    def score_window(self, window: List[str]) -> float:
        """
        Score a local context window (typically 5 words) and return perplexity.
        Handles padding with '<s>' and '</s>' tokens for boundary cases.
        
        Args:
            window: List of tokens forming the context window
            
        Returns:
            Perplexity score (lower is better fit to language model)
        """
        if len(window) == 0:
            return float('inf')
        
        log_prob_sum = 0.0
        num_ngrams = 0
        
        # Calculate log probability of n-grams in the window
        for i in range(len(window) - self.n + 1):
            ngram = tuple(window[i:i + self.n])
            prob = self.get_probability(ngram)
            
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                # Handle zero probability (shouldn't happen with smoothing)
                log_prob_sum += math.log(1e-10)
            
            num_ngrams += 1
        
        if num_ngrams == 0:
            return float('inf')
        
        # Perplexity = 2^(-log_prob / N)
        avg_log_prob = log_prob_sum / num_ngrams
        perplexity = math.pow(2, -avg_log_prob)
        
        return perplexity
