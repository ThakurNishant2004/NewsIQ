"""
qa_module.py
--------------
Answers user questions about a news article using prompt engineering
with a Large Language Model (LLM). Supports two pluggable providers —
Anthropic Claude and OpenAI GPT — selected via config.LLM_PROVIDER.

This module demonstrates prompt engineering techniques:
  1. Role assignment      -> tells the model to act as a "news analyst"
  2. Grounded context     -> injects the article text as the ONLY source
  3. Explicit constraints -> instructs the model to say "not mentioned in
                              the article" instead of hallucinating
  4. Output formatting    -> asks for concise, direct answers
"""

import logging
from typing import List, Dict

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a careful, factual news analyst assistant.
You will be given the full text of a news article and a user question.

Rules you MUST follow:
- Answer using ONLY information present in the article.
- If the article does not contain the answer, say clearly:
  "The article does not mention this."
- Do not invent facts, dates, names, or numbers.
- Keep answers concise (2-4 sentences) unless the user asks for detail.
- If useful, quote the exact relevant phrase from the article in quotes.
"""


def _build_user_prompt(article_text: str, question: str,
                        chat_history: List[Dict] = None) -> str:
    """
    Constructs the final user-turn prompt: article context + optional
    prior Q&A turns (for follow-up questions) + the new question.
    """
    history_block = ""
    if chat_history:
        turns = "\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history
        )
        history_block = f"\nPrevious conversation:\n{turns}\n"

    prompt = f"""ARTICLE:
\"\"\"
{article_text}
\"\"\"
{history_block}
QUESTION: {question}

Answer the question strictly based on the ARTICLE above."""
    return prompt


def _ask_anthropic(article_text: str, question: str,
                    chat_history: List[Dict] = None) -> str:
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    user_prompt = _build_user_prompt(article_text, question, chat_history)

    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.QA_MAX_TOKENS,
        temperature=config.QA_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _ask_openai(article_text: str, question: str,
                 chat_history: List[Dict] = None) -> str:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    user_prompt = _build_user_prompt(article_text, question, chat_history)

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=config.QA_TEMPERATURE,
        max_tokens=config.QA_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def answer_question(article_text: str, question: str,
                     chat_history: List[Dict] = None) -> str:
    """
    Main entry point. Routes to the configured LLM provider.
    `chat_history` is a list of {"question": ..., "answer": ...} dicts,
    allowing multi-turn follow-up questions about the same article.
    """
    if not article_text.strip() or not question.strip():
        return "Please provide both an article and a question."

    try:
        if config.LLM_PROVIDER == "anthropic":
            return _ask_anthropic(article_text, question, chat_history)
        elif config.LLM_PROVIDER == "openai":
            return _ask_openai(article_text, question, chat_history)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")
    except Exception as exc:
        logger.error("QA request failed: %s", exc)
        return (
            "Sorry, I couldn't reach the language model to answer that "
            f"question. (Error: {exc})"
        )


if __name__ == "__main__":
    demo_article = (
        "The Reserve Bank of India kept its key interest rate unchanged at "
        "6.5% on Thursday, citing easing inflation. Governor Shaktikanta Das "
        "said the next review will happen in December."
    )
    print(answer_question(demo_article, "What is the current interest rate?"))
