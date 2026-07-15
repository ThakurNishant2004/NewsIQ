"""
config.py
----------
Central place for model names, thresholds, and API configuration.
Keeping this separate makes it easy to swap models without touching
the logic in other modules (good software engineering practice).
"""




import os
from google import genai
from dotenv import load_dotenv


load_dotenv(override=True)


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


client = genai.Client(api_key=GEMINI_API_KEY)


# spaCy
SPACY_MODEL = "en_core_web_sm"


# Summarization
CHUNK_WORD_LIMIT = 500


QA_MAX_TOKENS = 400
QA_TEMPERATURE = 0.2


# Keyword Extraction
TOP_N_KEYWORDS = 10
TOP_N_ENTITIES = 5


# ----------------------- Misc -----------------------
RANDOM_SEED = 42
