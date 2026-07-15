"""
app.py
-------
Streamlit web interface for the Intelligent News Summarization and
Sentiment Analysis System. This is the entry point students run with:

    streamlit run app.py

It ties together preprocessing, summarization, sentiment analysis,
keyword/entity extraction, and the prompt-engineered QA module behind
a simple, interactive UI.
"""

import streamlit as st

from preprocessing import preprocess_pipeline
from summarizer import summarize_article
from sentiment_analysis import analyze_sentiment
from keyword_extraction import extract_all
from qa_module import answer_question
from utils import fetch_article_from_url, is_valid_url, word_count

st.set_page_config(
    page_title="Intelligent News Summarizer",
    page_icon="📰",
    layout="wide",
)

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "article_title" not in st.session_state:
    st.session_state.article_title = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ----------------------------------------------------------------------
# Sidebar: input controls
# ----------------------------------------------------------------------
st.sidebar.title("📰 News Input")
input_mode = st.sidebar.radio("Choose input method:", ["Paste text", "Paste URL"])

if input_mode == "Paste text":
    text_input = st.sidebar.text_area("Paste the article text here:", height=250)
    if st.sidebar.button("Load Article", use_container_width=True):
        if word_count(text_input) < 30:
            st.sidebar.error("Please paste a longer article (at least ~30 words).")
        else:
            st.session_state.article_text = text_input
            st.session_state.article_title = "Pasted Article"
            st.session_state.chat_history = []
            st.session_state.analysis_done = False

else:
    url_input = st.sidebar.text_input("Enter a news article URL:")
    if st.sidebar.button("Fetch Article", use_container_width=True):
        if not is_valid_url(url_input):
            st.sidebar.error("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Fetching article..."):
                try:
                    data = fetch_article_from_url(url_input)
                    st.session_state.article_text = data["text"]
                    st.session_state.article_title = data["title"]
                    st.session_state.chat_history = []
                    st.session_state.analysis_done = False
                except Exception as exc:
                    st.sidebar.error(str(exc))

use_transformer = st.sidebar.checkbox(
    "Use transformer models (uncheck for faster classical-NLP fallback)",
    value=True,
)

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("📰 Intelligent News Summarization & Sentiment Analysis")
st.caption(
    "NLP + LLM powered system for summarizing news, detecting sentiment, "
    "extracting entities, and answering questions about an article."
)

if not st.session_state.article_text:
    st.info("👈 Paste an article or a URL in the sidebar to get started.")
    st.stop()

st.subheader(st.session_state.article_title)
with st.expander("View original article text"):
    st.write(st.session_state.article_text)

run_analysis = st.button("🔍 Analyze Article", type="primary")

if run_analysis:
    st.session_state.analysis_done = True

if st.session_state.analysis_done:
    article_text = st.session_state.article_text

    tab_summary, tab_sentiment, tab_keywords, tab_qa, tab_debug = st.tabs(
        ["Summary", "Sentiment", "Keywords & Entities", "Ask Questions", "Preprocessing Debug"]
    )

    # ---------------- Summary tab ----------------
    with tab_summary:
        with st.spinner("Generating summary..."):
            summary = summarize_article(article_text, use_abstractive=use_transformer)
        st.markdown("### Generated Summary")
        st.write(summary)
        st.caption(f"Original: {word_count(article_text)} words → "
                   f"Summary: {word_count(summary)} words")

    # ---------------- Sentiment tab ----------------
    with tab_sentiment:
        with st.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment(article_text, use_transformer=use_transformer)

        label = sentiment["label"]
        emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}.get(label, "⚪")
        st.markdown(f"### Overall Sentiment: {emoji} {label}")
        st.progress(min(sentiment["score"], 1.0))
        st.caption(f"Confidence: {sentiment['score']:.2f}  |  Method: {sentiment['method']}")

    # ---------------- Keywords & Entities tab ----------------
    with tab_keywords:
        with st.spinner("Extracting keywords and entities..."):
            extracted = extract_all(article_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔑 Top Keywords (TF-IDF)")
            st.write(", ".join(extracted["tfidf_keywords"]) or "None found")
            st.markdown("### 🧩 Key Phrases (RAKE)")
            for phrase in extracted["rake_phrases"][:8]:
                st.markdown(f"- {phrase}")

        with col2:
            st.markdown("### 🏷️ Named Entities")
            if extracted["entities"]:
                for label_, values in extracted["entities"].items():
                    st.markdown(f"**{label_}**: {', '.join(values)}")
            else:
                st.write("No entities found (or spaCy model not installed).")

    # ---------------- QA tab ----------------
    with tab_qa:
        st.markdown("### 💬 Ask a question about this article")
        question = st.text_input("Your question:", key="qa_input")
        if st.button("Get Answer") and question.strip():
            with st.spinner("Thinking..."):
                answer = answer_question(
                    article_text, question, st.session_state.chat_history
                )
            st.session_state.chat_history.append(
                {"question": question, "answer": answer}
            )

        for turn in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {turn['question']}")
            st.markdown(f"**A:** {turn['answer']}")
            st.divider()

    # ---------------- Debug tab ----------------
    with tab_debug:
        st.markdown("### Preprocessing Output (for evaluation / viva)")
        debug_info = preprocess_pipeline(article_text)
        st.write(f"**Sentences:** {debug_info['num_sentences']}")
        st.write(f"**Words:** {debug_info['num_words']}")
        st.write("**First 30 lemmatized tokens:**")
        st.write(debug_info["lemmas"][:30])
