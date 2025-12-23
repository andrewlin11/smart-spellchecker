"""
Train an n-gram language model on larger corpora (Brown, Gutenberg, Reuters, WebText).
Outputs a pickled model file (default: language_model_large.pkl).
"""

import pickle
import sys
import os

# Add context_model to path
sys.path.insert(0, os.path.dirname(__file__))
from context_model import NGramLanguageModel

try:
    import nltk
    from nltk.corpus import brown, gutenberg, reuters, webtext
    nltk.data.find('corpora/brown')
    nltk.data.find('corpora/gutenberg')
    nltk.data.find('corpora/reuters')
    nltk.data.find('corpora/webtext')
except (ImportError, LookupError):
    print("Error: NLTK not installed or corpora not available.")
    print("Install with: pip install nltk")
    print("Download with: python -m nltk.downloader brown gutenberg reuters webtext")
    sys.exit(1)


def train_language_model(output_file: str = "language_model_large.pkl", n: int = 5):
    """
    Train an n-gram language model on multiple corpora.

    Args:
        output_file: Path to save the pickled model
        n: n-gram order (default 5)
    """
    print("Loading corpora: Brown, Gutenberg, Reuters, WebText...")
    try:
        corpora = [
            ("Brown", list(brown.words())),
            ("Gutenberg", list(gutenberg.words())),
            ("Reuters", list(reuters.words())),
            ("WebText", list(webtext.words())),
        ]

        total_words = 0
        all_words = []
        for name, words in corpora:
            print(f"  {name}: {len(words):,} words")
            total_words += len(words)
            all_words.extend(w.lower() for w in words)

        print(f"\nTotal words: {total_words:,}")
        training_text = ' '.join(all_words)
        print(f"Training corpus size: {len(training_text):,} characters")

    except Exception as e:
        print(f"Error loading corpora: {e}")
        sys.exit(1)

    print(f"\nTraining {n}-gram language model (n={n}, k=0.01)...")
    model = NGramLanguageModel(n=n, k=0.01)

    try:
        model.train(training_text)
        print("Model training complete!")
        print(f"  Vocabulary size: {model.vocab_size:,}")
        print(f"  Total n-grams: {len(model.ngram_counts):,}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    print(f"\nSaving model to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "language_model_large.pkl"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    train_language_model(output_file, n)
