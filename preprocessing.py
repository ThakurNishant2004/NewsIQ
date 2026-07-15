"""
preprocessing.py
------------------
Basic NLP preprocessing utilities used by every other module in the
pipeline: cleaning raw scraped text, sentence/word tokenization,
stopword removal, lemmatization, and chunking long articles so they
fit inside a transformer model's context window.
"""

import re
import string
import logging
from typing import List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources the first time the module is used."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", name)
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(raw_text: str) -> str:
    """
    Remove HTML leftovers, URLs, extra whitespace, and non-printable
    characters that commonly appear in scraped news articles.
    """
    if not raw_text:
        return ""

    text = re.sub(r"<[^>]+>", " ", raw_text)          # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # strip URLs
    text = re.sub(r"\s+", " ", text)                    # collapse whitespace
    text = text.strip()
    return text


def tokenize_sentences(text: str) -> List[str]:
    """Split cleaned text into a list of sentences."""
    return sent_tokenize(text)


def tokenize_words(text: str, remove_stopwords: bool = True,
                    remove_punctuation: bool = True) -> List[str]:
    """Split text into lowercase word tokens, optionally filtering
    stopwords and punctuation."""
    tokens = word_tokenize(text.lower())

    if remove_punctuation:
        tokens = [t for t in tokens if t not in string.punctuation]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    return tokens


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Reduce each token to its dictionary (lemma) form,
    e.g. 'running' -> 'run', 'better' -> 'good'."""
    return [_LEMMATIZER.lemmatize(token) for token in tokens]


def preprocess_pipeline(raw_text: str) -> dict:
    """
    Convenience function that runs the full preprocessing pipeline and
    returns intermediate results — useful for debugging and for the
    Streamlit UI's "show preprocessing steps" expander.
    """
    cleaned = clean_text(raw_text)
    sentences = tokenize_sentences(cleaned)
    word_tokens = tokenize_words(cleaned)
    lemmas = lemmatize_tokens(word_tokens)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "word_tokens": word_tokens,
        "lemmas": lemmas,
        "num_words": len(cleaned.split()),
        "num_sentences": len(sentences),
    }


def chunk_text(text: str, word_limit: int = 800) -> List[str]:
    """
    Split a long article into chunks of at most `word_limit` words,
    breaking on sentence boundaries so that no sentence is cut in half.
    This lets the summarizer / QA module handle articles longer than a
    transformer model's context window.
    """
    sentences = tokenize_sentences(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > word_limit and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [], 0
        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    sample = """
    NASA's Artemis II mission, announced today, will send four astronauts
    around the Moon in early 2026. Officials called it a "giant step" for
    future lunar landings. <p>Read more at http://example.com</p>
    """
    result = preprocess_pipeline(sample)
    print("Cleaned:", result["cleaned_text"])
    print("Sentences:", result["sentences"])
    print("Word count:", result["num_words"])
