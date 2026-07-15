# Intelligent News Summarization and Sentiment Analysis System

A final-year academic project that uses NLP and Large Language Models (LLMs) to
summarize news articles, detect sentiment, extract keywords/entities, and
answer user questions about an article using prompt engineering.

## 1. Problem Statement
Large volumes of online news make it difficult for users to quickly understand
important events. This system summarizes lengthy news articles, identifies
sentiment, extracts important entities/keywords, and answers user queries
related to the article — all through a simple web interface.

## 🔗 Live Demo

The app is deployed and live here: **[Try it out](https://thakurnishant2004-newsiq-app-mob4lo.streamlit.app/)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://thakurnishant2004-newsiq-app-mob4lo.streamlit.app/)

## 2. Project Architecture

```
news_summarizer_project/
│
├── app.py                  # Streamlit web interface (main entry point)
├── config.py                # Central configuration (model names, API keys)
├── preprocessing.py         # Text cleaning, tokenization, lemmatization
├── summarizer.py             # Abstractive + extractive summarization
├── sentiment_analysis.py     # Transformer + VADER sentiment detection
├── keyword_extraction.py     # TF-IDF / RAKE keywords + spaCy NER entities
├── qa_module.py               # Prompt-engineered Question Answering (LLM)
├── utils.py                    # Article fetching (URL), helper functions
├── requirements.txt
├── .env.example
└── sample_data/
    └── sample_article.txt
```

### High-level Pipeline
```
 Raw Article Text / URL
        │
        ▼
 preprocessing.py  ── cleans text, sentence/word tokenization
        │
        ├──► summarizer.py         ──► Summary (abstractive via BART / extractive via LexRank)
        ├──► sentiment_analysis.py ──► Sentiment label + confidence score
        ├──► keyword_extraction.py ──► Top keywords + Named Entities
        └──► qa_module.py          ──► Answers user questions (prompt engineering + LLM)
        │
        ▼
 app.py (Streamlit UI) ── displays everything interactively
```

## 3. Tech Stack
| Component | Library / Model |
|---|---|
| Preprocessing | NLTK, spaCy |
| Abstractive Summarization | HuggingFace Transformers (`facebook/bart-large-cnn`) |
| Extractive Summarization (fallback) | `sumy` (LexRank algorithm) |
| Sentiment Analysis | HuggingFace `distilbert-base-uncased-finetuned-sst-2-english` + VADER fallback |
| Keyword Extraction | scikit-learn TF-IDF + RAKE (`rake-nltk`) |
| Named Entity Recognition | spaCy `en_core_web_sm` |
| Question Answering | Prompt-engineered call to an LLM (Anthropic Claude / OpenAI GPT, pluggable) |
| Web Interface | Streamlit |
| Article Fetching from URL | `newspaper3k`, `requests`, `beautifulsoup4` |

## 4. Setup Instructions

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download required NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

# 4. Configure API keys
cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 5. Run the app
streamlit run app.py
```

The first run will download the BART and DistilBERT models from HuggingFace
(a few hundred MB) — an internet connection is required for that one-time
download. After that, the models are cached locally.

## 5. Features Implemented (mapped to objectives)
- [x] NLP preprocessing (cleaning, tokenization, stopword removal, lemmatization)
- [x] Abstractive summarization with automatic extractive fallback
- [x] Sentiment detection (transformer-based, with rule-based fallback)
- [x] Keyword extraction (TF-IDF + RAKE) and Named Entity Recognition
- [x] Prompt-engineered Question-Answering over the article (LLM)
- [x] Interactive Streamlit web interface (upload text, paste text, or paste URL)

## 6. Notes for Evaluation / Viva
- The system is **modular**: each capability lives in its own file and can be
  unit-tested independently (see the `if __name__ == "__main__":` demo block
  at the bottom of every module).
- Summarization and sentiment both have **graceful fallbacks** — if the
  transformer model cannot be loaded (e.g., no internet, low RAM), the system
  automatically switches to lightweight classical NLP methods so the app
  never crashes.
- The QA module demonstrates **prompt engineering**: it constructs a
  structured prompt (role + context + constraints + question) rather than
  simply concatenating the question with the article.
- Known limitations: extremely long articles are chunked before
  summarization/QA because transformer models have a limited context window;
  see `preprocessing.chunk_text()`.

## 7. Future Scope
- Multi-language support
- Comparative sentiment across multiple articles on the same event
- Bias/fact-check detection layer
- Caching + async processing for faster response on repeated queries
