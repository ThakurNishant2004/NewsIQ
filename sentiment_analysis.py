"""
sentiment_analysis.py
-----------------------
Detects the overall sentiment (Positive / Negative / Neutral) of a
news article, along with a confidence score.

Primary method : Transformer-based classifier (DistilBERT fine-tuned
                  on SST-2) via HuggingFace `transformers`.
Fallback method : VADER (Valence Aware Dictionary and sEntiment
                  Reasoner) from NLTK — a lightweight rule-based
                  sentiment analyzer that works well on short text
                  and requires no model download.
"""

import logging
from typing import Dict, List

import nltk
import config
from preprocessing import tokenize_sentences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_sentiment_pipeline = None


def _load_sentiment_model():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        logger.info("Loading sentiment model: %s", config.SENTIMENT_MODEL)
        _sentiment_pipeline = pipeline(
            "sentiment-analysis", model=config.SENTIMENT_MODEL
        )
    return _sentiment_pipeline


def _vader_sentiment(text: str) -> Dict:
    """Fallback sentiment scoring using VADER (no heavy model needed)."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)

    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {"label": label, "score": abs(compound), "method": "vader"}


def analyze_sentiment(text: str, use_transformer: bool = True) -> Dict:
    """
    Analyze the sentiment of a piece of text (typically the article
    summary, since transformer models have a max input length of ~512
    tokens). Returns a dict: {label, score, method}.
    """
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0, "method": "none"}

    if use_transformer:
        try:
            model = _load_sentiment_model()
            # Transformer models cap input at 512 tokens; truncate defensively.
            result = model(text[:2000], truncation=True)[0]
            return {
                "label": result["label"].upper(),
                "score": round(float(result["score"]), 3),
                "method": "transformer",
            }
        except Exception as exc:
            logger.warning("Transformer sentiment failed (%s). "
                            "Falling back to VADER.", exc)

    return _vader_sentiment(text)


def analyze_sentence_level(text: str, use_transformer: bool = True) -> List[Dict]:
    """
    Break the article into sentences and return per-sentence sentiment.
    Useful for showing a sentiment timeline / breakdown chart in the UI.
    """
    sentences = tokenize_sentences(text)
    return [
        {"sentence": s, **analyze_sentiment(s, use_transformer)}
        for s in sentences
    ]


if __name__ == "__main__":
    sample = ("Investors cheered the surprise earnings beat, sending shares "
              "soaring more than 12% in after-hours trading.")
    print(analyze_sentiment(sample))
