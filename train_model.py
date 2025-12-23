"""
Train an n-gram language model on the Brown Corpus for spell checking.
This script downloads the Brown Corpus from NLTK, trains a trigram model
with add-k smoothing, and saves it to language_model.pkl for later use.
"""

import pickle
import sys
import os

# Add context_model to path
sys.path.insert(0, os.path.dirname(__file__))
from context_model import NGramLanguageModel

try:
    import nltk
    # Ensure corpora are available
    from nltk.corpus import brown, gutenberg, reuters
    nltk.data.find('corpora/brown')
    nltk.data.find('corpora/gutenberg')
    nltk.data.find('corpora/reuters')
except (ImportError, LookupError):
    print("Error: NLTK not installed or corpora not available.")
    print("Install with: pip install nltk")
    print("Download corpora with: python -m nltk.downloader brown gutenberg reuters")
    sys.exit(1)


def train_language_model(output_file: str = "language_model.pkl"):
    """
    Train a trigram language model on the Brown Corpus.
    
    Args:
        output_file: Path to save the pickled model
    """
    print("Loading Brown + Gutenberg + Reuters corpora...")
    try:
        # Get words from corpora
        brown_words = list(brown.words())
        gutenberg_words = list(gutenberg.words())
        reuters_words = list(reuters.words())
        total_words = len(brown_words) + len(gutenberg_words) + len(reuters_words)
        print(
            f"Loaded {len(brown_words)} (Brown) + {len(gutenberg_words)} (Gutenberg) + {len(reuters_words)} (Reuters) = {total_words} words"
        )

        words_lower = [w.lower() for w in brown_words] + [w.lower() for w in gutenberg_words] + [w.lower() for w in reuters_words]
        # Join into a single text string for training
        training_text = ' '.join(words_lower)
        print(f"Training corpus size: {len(training_text)} characters")
        
    except Exception as e:
        print(f"Error loading Brown Corpus: {e}")
        print("Make sure you have downloaded the Brown Corpus.")
        print("Run: python -m nltk.downloader brown")
        sys.exit(1)
    
    # Create and train the model
    print("\nTraining 5-gram language model (n=5, k=0.01)...")
    model = NGramLanguageModel(n=5, k=0.01)
    
    try:
        model.train(training_text)
        print(f"Model training complete!")
        print(f"  Vocabulary size: {model.vocab_size}")
        print(f"  Total n-grams: {len(model.ngram_counts)}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    # Save the trained model
    print(f"\nSaving model to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully!")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    output_file = "language_model.pkl" if len(sys.argv) < 2 else sys.argv[1]
    train_language_model(output_file)
