"""
Load and parse Birkbeck spelling error corpus from a local file.
Format: $ indicates correct word, words below are misspellings, _ indicates space.
"""

import pickle
import sys
from typing import Dict, Set


def load_birkbeck_corpus(filepath: str) -> Dict[str, Set[str]]:
    """
    Load Birkbeck spelling error corpus from a text file.
    
    Format:
        $correct_word
        misspelling1
        misspelling2
        ...
    
    Where _ represents a space in the word.
    
    Args:
        filepath: Path to the Birkbeck corpus file
        
    Returns:
        Dictionary mapping misspellings to set of possible correct words
        {misspelling: {correct_word1, correct_word2, ...}}
    """
    error_dict = {}
    current_correct = None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    # Skip empty lines
                    continue
                
                if line.startswith('$'):
                    # This is a correct word
                    current_correct = line[1:]  # Remove the '$' prefix
                    # Replace underscores with spaces
                    current_correct = current_correct.replace('_', ' ')
                else:
                    # This is a misspelling
                    if current_correct is not None:
                        misspelling = line.replace('_', ' ')
                        
                        if misspelling not in error_dict:
                            error_dict[misspelling] = set()
                        
                        error_dict[misspelling].add(current_correct)
        
        return error_dict
        
    except FileNotFoundError:
        print(f"Error: Could not find Birkbeck corpus file at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Birkbeck corpus: {e}")
        sys.exit(1)


def save_error_patterns(error_dict: Dict[str, Set[str]], output_file: str = "error_patterns.pkl"):
    """
    Save the error patterns dictionary to a pickle file.
    
    Args:
        error_dict: Dictionary of misspellings to correct words
        output_file: Path to save the pickle file
    """
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(error_dict, f)
        print(f"Error patterns saved to {output_file}")
    except Exception as e:
        print(f"Error saving error patterns: {e}")
        sys.exit(1)


def main():
    """Main function to load Birkbeck corpus and save to pickle."""
    if len(sys.argv) < 2:
        print("Usage: python error_patterns.py <birkbeck_corpus.txt> [output_file.pkl]")
        sys.exit(1)
    
    corpus_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "error_patterns.pkl"
    
    print(f"Loading Birkbeck corpus from {corpus_file}...")
    error_dict = load_birkbeck_corpus(corpus_file)
    
    print(f"Loaded {len(error_dict)} unique misspellings")
    
    # Print some statistics
    total_mappings = sum(len(correct_words) for correct_words in error_dict.values())
    print(f"Total misspelling-to-correct mappings: {total_mappings}")
    
    # Show a few examples
    print("\nExample entries:")
    for i, (misspelling, correct_words) in enumerate(list(error_dict.items())[:5]):
        print(f"  '{misspelling}' → {correct_words}")
    
    # Save to pickle
    save_error_patterns(error_dict, output_file)


if __name__ == "__main__":
    main()
