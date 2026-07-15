"""
keyword_extraction.py
------------------------
Extracts the most important keywords/keyphrases from a news article
(using TF-IDF and RAKE) and identifies named entities such as people,
organizations, locations, and dates (using spaCy NER).
"""

import logging
from typing import Dict, List

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_nlp = None  # lazy-loaded spaCy model


def _load_spacy_model():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(config.SPACY_MODEL)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found locally. Run: "
                "python -m spacy download %s",
                config.SPACY_MODEL, config.SPACY_MODEL,
            )
            raise
    return _nlp


def extract_keywords_tfidf(text: str, top_n: int = config.TOP_N_KEYWORDS) -> List[str]:
    """
    Extract top keywords using TF-IDF over the article's sentences
    (each sentence treated as a 'document' so common words that appear
    everywhere get a lower score).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from preprocessing import tokenize_sentences

    sentences = tokenize_sentences(text)
    if len(sentences) < 2:
        sentences = [text, text]  # TF-IDF needs at least 2 "documents"

    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = tfidf_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

    return [term for term, _ in ranked[:top_n]]


def extract_keywords_rake(text: str, top_n: int = config.TOP_N_KEYWORDS) -> List[str]:
    """Extract keyphrases using RAKE (Rapid Automatic Keyword Extraction),
    which is good at pulling out multi-word phrases (e.g. 'interest rate hike')
    rather than single words."""
    from rake_nltk import Rake

    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:top_n]


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities grouped by type (PERSON, ORG, GPE/location,
    DATE, MONEY, etc.) using spaCy's pretrained NER model.
    """
    try:
        nlp = _load_spacy_model()
    except OSError:
        return {}

    doc = nlp(text)
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)

    return entities


def extract_all(text: str) -> Dict:
    """Convenience wrapper returning keywords (TF-IDF + RAKE) and entities together."""
    return {
        "tfidf_keywords": extract_keywords_tfidf(text),
        "rake_phrases": extract_keywords_rake(text),
        "entities": extract_entities(text),
    }


if __name__ == "__main__":
    sample = (
        "Apple CEO Tim Cook announced on Monday that the company will invest "
        "$500 million in a new manufacturing plant in Bengaluru, India, "
        "creating over 2,000 jobs by 2027."
    )
    result = extract_all(sample)
    print("TF-IDF keywords:", result["tfidf_keywords"])
    print("RAKE phrases:", result["rake_phrases"])
    print("Entities:", result["entities"])
