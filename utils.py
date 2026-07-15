"""
utils.py
---------
Helper functions that don't belong to a specific NLP stage:
fetching article text from a URL, and basic input validation.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_article_from_url(url: str) -> dict:
    """
    Download and extract the main article text + title from a news URL
    using `newspaper3k`. Returns {"title": str, "text": str} or raises
    a ValueError with a friendly message on failure.
    """
    try:
        from newspaper import Article
    except ImportError as exc:
        raise ImportError(
            "newspaper3k is not installed. Run: pip install newspaper3k"
        ) from exc

    try:
        article = Article(url)
        article.download()
        article.parse()
    except Exception as exc:
        logger.error("Failed to fetch article from %s: %s", url, exc)
        raise ValueError(
            f"Could not fetch or parse the article at this URL. ({exc})"
        )

    if not article.text.strip():
        raise ValueError("No readable article text found at this URL.")

    return {"title": article.title, "text": article.text}


def is_valid_url(text: str) -> bool:
    """Quick heuristic check for whether the given input is a URL."""
    return text.strip().lower().startswith(("http://", "https://"))


def word_count(text: str) -> int:
    return len(text.split())


if __name__ == "__main__":
    print(is_valid_url("https://www.bbc.com/news/some-article"))
    print(is_valid_url("This is just plain article text."))
