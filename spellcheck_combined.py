"""
Combined spell checker using neural LM, Birkbeck corpus, edit distance, and n-gram model.
Priority: Birkbeck > Neural Model > Show 1 edit-distance + 1 n-gram suggestion.
"""

import sys
import re
import pickle
from edit_distance_rec import distance_fast
from context_model import NGramLanguageModel

# Patterns for filtering
ACRONYM_PATTERN = r'^(?:[A-Z]\.)+[A-Z]?\.?$'
PUNCTUATION_PATTERN = r'^[\[\].,;"\'()?_:`\-]+$'
CONTRACTION_PATTERN = r"\w+['']\w+"

def simple_tokenize(text):
    """Simple whitespace-based tokenizer."""
    return text.split()

def should_check(token):
    """Check if a token should be spell-checked."""
    if re.match(PUNCTUATION_PATTERN, token):
        return False
    if re.match(CONTRACTION_PATTERN, token):
        return False
    if re.match(ACRONYM_PATTERN, token):
        return False
    if any(char.isdigit() for char in token):
        return False
    if not any(char.isalpha() for char in token):
        return False
    return True

def strip_punctuation(token):
    """Strip leading and trailing punctuation from a token."""
    leading = ""
    trailing = ""

    while token and not token[0].isalpha():
        leading += token[0]
        token = token[1:]

    while token and not token[-1].isalpha():
        trailing = token[-1] + trailing
        token = token[:-1]

    return token, leading, trailing

def find_edit_distance_candidates(misspelled, correct_words_dict, max_dist=2, limit=2):
    """Find top N candidates by edit distance."""
    candidates = []
    word_len = len(misspelled)

    for correct_word in correct_words_dict:
        if abs(len(correct_word) - word_len) > max_dist + 1:
            continue

        dist = distance_fast(misspelled.lower(), correct_word.lower())

        if dist <= max_dist:
            candidates.append((dist, correct_word))

    candidates.sort()
    return candidates[:limit]

class NeuralLanguageModel:
    """Wrapper for transformer-based language models (lazy-loaded)."""

    def __init__(self, model_name="distilbert-base-uncased"):
        """Initialize neural LM from Hugging Face (lazy loading)."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.device = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch
        except ImportError:
            print("Error: transformers not installed")
            print("Install with: pip install transformers torch")
            sys.exit(1)

        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        self._loaded = True

    def score_candidate_in_context(self, candidate, tokens, position):
        """Score a candidate word at a position using masked LM (lower is better)."""
        self._ensure_loaded()
        
        sentence_with_mask = ' '.join(tokens[:position] + ['[MASK]'] + tokens[position + 1:])
        masked_inputs = self.tokenizer(sentence_with_mask, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**masked_inputs)
            predictions = outputs.logits

        mask_token_index = (masked_inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_token_index) == 0:
            return float('inf')
        mask_token_index = mask_token_index[0]

        candidate_token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
        if not candidate_token_ids:
            return float('inf')
        candidate_token_id = candidate_token_ids[0]

        logits = predictions[0, mask_token_index]
        log_probs = self.torch.nn.functional.log_softmax(logits, dim=0)
        score = -log_probs[candidate_token_id].item()
        return score

def load_dictionary(fname):
    """Load dictionary as set for O(1) lookups."""
    with open(fname, 'r') as fp:
        return set(line.strip().lower() for line in fp)

def load_error_patterns(fname="error_patterns.pkl"):
    """Load error patterns from pickle file."""
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def load_ngram_model(fname="language_model.pkl"):
    """Load n-gram language model from pickle file."""
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: N-gram model '{fname}' not found. Run train_model.py first.")
        return None

def spellcheck_combined(token, position, tokens, correct_words, neural_lm, ngram_lm, error_patterns):
    """
    Combined spell-check with priority:
    1. If Birkbeck has a replacement, use it (best suggestion)
    2. Otherwise, use neural LM prediction (best suggestion)
    3. Show alternatives: 1 from edit-distance + 1 from n-gram model
    """
    if token.lower() in correct_words:
        return None, None, None, []

    # Priority 1: Check Birkbeck
    birk_candidates = list(error_patterns.get(token.lower(), [])) if error_patterns else []
    
    # Get edit distance candidates
    ed_candidates = find_edit_distance_candidates(token, correct_words, max_dist=2, limit=10)
    ed_words = [c for _, c in ed_candidates]
    
    # Get n-gram model suggestion
    ngram_suggestion = None
    if ngram_lm and ed_words:
        # Score candidates with n-gram model
        ngram_scored = []
        for candidate in ed_words:
            test_tokens = tokens.copy()
            test_tokens[position] = candidate.lower()
            sentence_text = ' '.join(test_tokens)
            perplexity = ngram_lm.get_perplexity(sentence_text)
            ngram_scored.append((perplexity, candidate))
        
        ngram_scored.sort()
        if ngram_scored:
            ngram_suggestion = ngram_scored[0][1]
    
    # Get top edit-distance suggestion
    ed_suggestion = ed_words[0] if ed_words else None
    
    if birk_candidates:
        best_suggestion = birk_candidates[0]
        return best_suggestion, ed_suggestion, ngram_suggestion, ["(Birkbeck)"]

    # Priority 2: Use neural LM
    if not ed_candidates:
        return None, None, None, []

    # Score with neural LM
    scored = []
    for candidate in ed_words:
        score = neural_lm.score_candidate_in_context(candidate, tokens, position)
        scored.append((score, candidate))

    scored.sort()
    if not scored:
        return None, None, None, []

    best_suggestion = scored[0][1]
    return best_suggestion, ed_suggestion, ngram_suggestion, ["(Neural LM)"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python spellcheck_combined.py filename [model_name]")
        print("\nAvailable models:")
        print("  - distilbert-base-uncased (default, fast)")
        print("  - bert-base-uncased")
        print("  - roberta-base")
        sys.exit(1)

    model_name = sys.argv[2] if len(sys.argv) > 2 else "distilbert-base-uncased"

    with open(sys.argv[1]) as fp:
        full_text = fp.read()

    tokens = simple_tokenize(full_text)

    print("Loading resources...")
    neural_lm = NeuralLanguageModel(model_name)

    print("Loading n-gram model...")
    ngram_lm = load_ngram_model()

    print("Loading error patterns...")
    error_patterns = load_error_patterns()

    print("Loading dictionary...")
    correct_words = load_dictionary("words_alpha.txt")

    print("Spell-checking...\n")

    for i, token in enumerate(tokens):
        clean_token, leading, trailing = strip_punctuation(token)

        if clean_token and should_check(clean_token):
            best, ed_alt, ngram_alt, source = spellcheck_combined(
                clean_token, i, tokens, correct_words, neural_lm, ngram_lm, error_patterns
            )

            if best:
                print(f"{clean_token} -> {best} {source[0]}")
                alternatives = []
                if ed_alt:
                    alternatives.append(f" {ed_alt} (edit-distance)")
                if ngram_alt:
                    alternatives.append(f" {ngram_alt} (n-gram)")
                if alternatives:
                    print(f"  alternatives: {', '.join(alternatives)}")

if __name__ == '__main__':
    main()
