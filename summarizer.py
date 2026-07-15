"""
summarizer.py
--------------
Generates concise summaries of news articles.

Primary method : Abstractive summarization using a pretrained
                  transformer (facebook/bart-large-cnn) via HuggingFace
                  `transformers` pipeline.
Fallback method : Extractive summarization using the LexRank algorithm
                  (via the `sumy` library). This runs even if the
                  transformer model can't be downloaded/loaded, so the
                  app degrades gracefully instead of crashing.
"""

import logging
from typing import List

import config
from preprocessing import chunk_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_abstractive_pipeline = None  # lazy-loaded, cached after first use


def _load_abstractive_model():
    """Lazily load the HuggingFace summarization pipeline (heavy import)."""
    global _abstractive_pipeline
    if _abstractive_pipeline is None:
        from transformers import pipeline
        logger.info("Loading abstractive summarization model: %s",
                    config.ABSTRACTIVE_SUMMARY_MODEL)
        _abstractive_pipeline = pipeline(
            "summarization", model=config.ABSTRACTIVE_SUMMARY_MODEL
        )
    return _abstractive_pipeline


def _extractive_summary(text: str, num_sentences: int = 4) -> str:
    """
    Fallback extractive summarizer using LexRank. Picks the most
    'central' sentences of the article rather than generating new text.
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary_sentences)


def summarize_chunk(text: str, use_abstractive: bool = True) -> str:
    """Summarize a single chunk of text (must fit the model's context window)."""
    if not text.strip():
        return ""

    if use_abstractive:
        try:
            model = _load_abstractive_model()
            output = model(
                text,
                max_length=config.MAX_SUMMARY_LENGTH,
                min_length=config.MIN_SUMMARY_LENGTH,
                do_sample=False,
            )
            return output[0]["summary_text"]
        except Exception as exc:  # model missing, OOM, no internet, etc.
            logger.warning("Abstractive summarization failed (%s). "
                            "Falling back to extractive method.", exc)

    return _extractive_summary(text)


def summarize_article(text: str, use_abstractive: bool = True) -> str:
    """
    Full pipeline: chunk the article if it's too long, summarize each
    chunk, and if multiple chunks were produced, run a second summarization
    pass over the concatenated chunk-summaries ("map-reduce" summarization).
    """
    chunks: List[str] = chunk_text(text, word_limit=config.CHUNK_WORD_LIMIT)

    if len(chunks) == 1:
        return summarize_chunk(chunks[0], use_abstractive)

    logger.info("Article split into %d chunks; using map-reduce summarization.",
                len(chunks))
    partial_summaries = [summarize_chunk(c, use_abstractive) for c in chunks]
    combined = " ".join(partial_summaries)

    # Second pass to compress the combined chunk-summaries into one
    # coherent final summary.
    return summarize_chunk(combined, use_abstractive)


if __name__ == "__main__":
    sample_article = (
        "The Reserve Bank of India kept its key interest rate unchanged "
        "for the third consecutive meeting on Thursday, citing easing "
        "inflation but persistent global uncertainty. The central bank's "
        "governor said the decision reflects a 'wait and watch' approach "
        "as the economy navigates external headwinds including volatile "
        "crude oil prices and geopolitical tensions. Analysts had largely "
        "expected the pause, with most brokerages predicting the next rate "
        "cut could come only in the final quarter of the fiscal year. "
        "Stock markets reacted positively to the announcement, with the "
        "benchmark index closing 0.8% higher."
    )
    print("SUMMARY:\n", summarize_article(sample_article))
