import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def extract_features(text):
    text = str(text).strip()

    # Extract words
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())

    # Extract sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    word_lengths = [len(word) for word in words]

    word_counts = Counter(words)
    unique_words = len(set(words))

    # Words that occur only once
    hapax_count = sum(
        1 for count in word_counts.values() if count == 1
    )

    # Number of words in each sentence
    sentence_lengths = [
        len(re.findall(r"\b[a-zA-Z']+\b", sentence))
        for sentence in sentences
    ]

    # Stopwords
    stopword_count = sum(
        1 for word in words if word in ENGLISH_STOP_WORDS
    )

    # Punctuation
    punctuation_count = len(re.findall(r"[,;:!?\.]", text))

    return {
        "average_word_length": (
            np.mean(word_lengths) if word_lengths else 0
        ),

        "average_sentence_length": (
            word_count / sentence_count if sentence_count else 0
        ),

        "sentence_length_std": (
            np.std(sentence_lengths) if sentence_lengths else 0
        ),

        "type_token_ratio": (
            unique_words / word_count if word_count else 0
        ),

        "hapax_ratio": (
            hapax_count / word_count if word_count else 0
        ),

        "stopword_ratio": (
            stopword_count / word_count if word_count else 0
        ),

        "punctuation_per_100_words": (
            punctuation_count / word_count * 100 if word_count else 0
        )
    }