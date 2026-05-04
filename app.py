import streamlit as st
import os
import re
import html
import time
import random
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="YT Insight — AI Video Chat",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0A0F !important;
    color: #E8E6F0 !important;
}

.stApp {
    background: #0A0A0F !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0A0F; }
::-webkit-scrollbar-thumb { background: #2D2A40; border-radius: 2px; }

.yt-hero {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.yt-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at 50% 0%, rgba(255, 45, 85, 0.12) 0%, transparent 70%);
    pointer-events: none;
}
.yt-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 45, 85, 0.1);
    border: 1px solid rgba(255, 45, 85, 0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FF6B8A;
    margin-bottom: 1.2rem;
}
.yt-badge-dot {
    width: 6px; height: 6px;
    background: #FF2D55;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}
.yt-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 400;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #F0EEF8;
    margin-bottom: 0.75rem;
}
.yt-title em {
    font-style: italic;
    color: #FF6B8A;
}
.yt-subtitle {
    font-size: 1rem;
    color: #7A7890;
    font-weight: 300;
    letter-spacing: 0.01em;
    max-width: 440px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
    text-align: center !important;
    width: 100%;
    display: block;
}

/* Force Streamlit's own markdown containers to center too */
.yt-hero p,
.yt-hero .yt-subtitle,
[data-testid="stMarkdownContainer"] .yt-subtitle,
[data-testid="stMarkdownContainer"] p {
    text-align: center !important;
    width: 100% !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.yt-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2D2A40 30%, #2D2A40 70%, transparent);
    margin: 0 0 2rem;
}

.yt-panel {
    background: #12111A;
    border: 1px solid #1E1C2E;
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.yt-panel::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
}

.yt-label {
    font-size: 10.5px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4A4860;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.yt-label::before {
    content: '';
    display: inline-block;
    width: 16px; height: 1px;
    background: #4A4860;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0D0C16 !important;
    border: 1px solid #1E1C2E !important;
    border-radius: 10px !important;
    color: #E8E6F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #FF2D55 !important;
    box-shadow: 0 0 0 3px rgba(255, 45, 85, 0.1) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #3A3850 !important;
}

.stSelectbox > div > div {
    background: #0D0C16 !important;
    border: 1px solid #1E1C2E !important;
    border-radius: 10px !important;
    color: #E8E6F0 !important;
    cursor: pointer !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #FF2D55 !important;
    box-shadow: 0 0 0 3px rgba(255, 45, 85, 0.1) !important;
}

/* Pointer cursor on all interactive parts of the selectbox */
.stSelectbox,
.stSelectbox *,
.stSelectbox > div,
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] * {
    cursor: pointer !important;
}

.stButton > button {
    background: linear-gradient(135deg, #FF2D55, #C70039) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s, transform 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

.stTextInput label, .stTextArea label, .stSelectbox label {
    color: #4A4860 !important;
    font-size: 10.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

.answer-card {
    background: #0E0D18;
    border: 1px solid #1E1C2E;
    border-left: 3px solid #FF2D55;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    animation: slide-in 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes slide-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.answer-tag {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #FF6B8A;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.answer-tag::before {
    content: '';
    width: 8px; height: 8px;
    background: #FF2D55;
    border-radius: 50%;
}
.answer-text {
    font-size: 1rem;
    line-height: 1.75;
    color: #C8C6D8;
    font-weight: 300;
}

.stSpinner > div {
    border-top-color: #FF2D55 !important;
}

.stAlert {
    background: rgba(199, 0, 57, 0.08) !important;
    border: 1px solid rgba(255, 45, 85, 0.2) !important;
    border-radius: 10px !important;
    color: #FF6B8A !important;
}

.yt-footer {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    color: #2D2A40;
    font-size: 12px;
    letter-spacing: 0.05em;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 11px;
    color: #4ade80;
    font-weight: 500;
    margin-bottom: 1.25rem;
}
.status-dot {
    width: 5px; height: 5px;
    background: #22c55e;
    border-radius: 50%;
}

#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 680px !important; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="yt-hero">
    <div class="yt-badge">
        <span class="yt-badge-dot"></span>
        AI-Powered · Instant Insights
    </div>
    <h1 class="yt-title">Chat with any<br><em>YouTube video</em></h1>
    <p class="yt-subtitle">Paste a video link, ask anything — get precise answers drawn straight from the transcript.</p>
</div>
<div class="yt-divider"></div>
""", unsafe_allow_html=True)

st.text_input(
    "VIDEO URL",
    placeholder="https://youtube.com/watch?v=...",
    key="video_url",
    label_visibility="visible",
)
col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    language = st.selectbox(
        "LANGUAGE",
        ["English", "Hindi"],
        label_visibility="visible",
    )

with col2:
    query = st.text_input(
        "YOUR QUESTION",
        placeholder="What is this video about?",
        label_visibility="visible",
    )
ask = st.button("Ask AI →")

@st.cache_resource
def process_video(url):
    try:
        video_id = extract_video_id(url)
        transcript_text = load_transcript_text(video_id)
        if not transcript_text:
            return None, "Transcript text was empty for this video."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(transcript_text)

        if not chunks:
            return None, "Transcript could not be split into searchable chunks."

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store, None
    except Exception as error:
        return None, str(error)


def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    if parsed_url.hostname and "youtube.com" in parsed_url.hostname:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [""])[0]

        if parsed_url.path.startswith(("/shorts/", "/embed/")):
            parts = parsed_url.path.split("/")
            return parts[2] if len(parts) > 2 else ""

    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else ""


def load_transcript_text(video_id):
    if not video_id:
        raise ValueError("Please provide a valid YouTube URL.")

    cooldown_error = get_rate_limit_cooldown_error(video_id)
    if cooldown_error:
        raise ValueError(cooldown_error)

    try:
        return load_transcript_text_from_api(video_id)
    except Exception as primary_error:
        try:
            return load_transcript_text_with_ytdlp(video_id)
        except Exception as fallback_error:
            if is_rate_limit_error(primary_error) or is_rate_limit_error(fallback_error):
                set_rate_limit_cooldown(video_id)
                raise ValueError(
                    "YouTube is rate limiting requests (HTTP 429). Please wait 1-2 minutes and try again."
                ) from fallback_error
            raise


def get_rate_limit_cooldown_error(video_id):
    cooldowns = st.session_state.get("yt_rate_limit_cooldowns", {})
    cooldown_until = cooldowns.get(video_id)

    if not cooldown_until:
        return None

    remaining_seconds = int(cooldown_until - time.time())
    if remaining_seconds <= 0:
        cooldowns.pop(video_id, None)
        st.session_state["yt_rate_limit_cooldowns"] = cooldowns
        return None

    remaining_minutes = max(1, (remaining_seconds + 59) // 60)
    return f"YouTube is rate limiting requests (HTTP 429). Please wait about {remaining_minutes} minute(s) and try again."


def set_rate_limit_cooldown(video_id, cooldown_seconds=120):
    cooldowns = st.session_state.get("yt_rate_limit_cooldowns", {})
    cooldowns[video_id] = time.time() + cooldown_seconds
    st.session_state["yt_rate_limit_cooldowns"] = cooldowns


def is_rate_limit_error(error):
    if isinstance(error, HTTPError) and error.code == 429:
        return True
    return "429" in str(error)


def backoff_delay_seconds(attempt):
    return (2 ** attempt) + random.uniform(0, 0.5)


def load_transcript_text_from_api(video_id, max_attempts=4):
    last_error = None

    for attempt in range(max_attempts):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            transcript = None
            try:
                transcript = transcript_list.find_transcript(["en", "hi"])
            except Exception:
                for candidate in transcript_list:
                    transcript = candidate
                    break

            if transcript is None:
                raise ValueError("No transcript track was found for this video.")

            transcript_chunks = transcript.fetch()
            text = " ".join(chunk["text"] for chunk in transcript_chunks if chunk.get("text"))
            if text.strip():
                return text
            raise ValueError("Transcript text was empty after fetching captions.")
        except Exception as error:
            last_error = error
            if is_rate_limit_error(error) and attempt < max_attempts - 1:
                time.sleep(backoff_delay_seconds(attempt))
                continue
            raise

    raise last_error


def load_transcript_text_with_ytdlp(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "hi"],
    }

    info = None
    last_error = None
    for attempt in range(4):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
            break
        except Exception as error:
            last_error = error
            if is_rate_limit_error(error) and attempt < 3:
                time.sleep(backoff_delay_seconds(attempt))
                continue
            raise

    if info is None and last_error is not None:
        raise last_error

    subtitle_sets = [info.get("subtitles") or {}, info.get("automatic_captions") or {}]

    for subtitle_set in subtitle_sets:
        subtitle_url = pick_subtitle_url(subtitle_set)
        if subtitle_url:
            return subtitle_url_to_text(subtitle_url)

    raise ValueError("No usable subtitle track was found for this video.")


def pick_subtitle_url(subtitle_set):
    preferred_languages = ["en", "en-US", "en-GB", "hi"]

    for language_code in preferred_languages:
        tracks = subtitle_set.get(language_code)
        if tracks:
            return tracks[0].get("url")

    for tracks in subtitle_set.values():
        if tracks:
            return tracks[0].get("url")

    return ""


def subtitle_url_to_text(subtitle_url):
    request = urllib.request.Request(
        subtitle_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    subtitle_content = ""
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request) as response:
                subtitle_content = response.read().decode("utf-8", errors="ignore")
            break
        except Exception as error:
            last_error = error
            if is_rate_limit_error(error) and attempt < 3:
                time.sleep(backoff_delay_seconds(attempt))
                continue
            raise

    if not subtitle_content and last_error is not None:
        raise last_error

    cleaned_lines = []
    for line in subtitle_content.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line == "WEBVTT":
            continue
        if stripped_line.isdigit():
            continue
        if "-->" in stripped_line:
            continue
        if stripped_line.startswith("NOTE"):
            continue

        cleaned_line = re.sub(r"<[^>]+>", "", stripped_line)
        cleaned_lines.append(html.unescape(cleaned_line))

    transcript_text = " ".join(cleaned_lines)
    if not transcript_text.strip():
        raise ValueError("Subtitle track was found but no text could be extracted.")

    return transcript_text


video_url = st.session_state.get("video_url", "")

if video_url:
    with st.spinner("Loading transcript…"):
        vector_store, transcript_error = process_video(video_url)

    if vector_store is None:
        st.error(transcript_error or "Transcript unavailable for this video. Check the URL and try again.")
        st.stop()

    st.markdown("""
    <div class="status-pill">
        <span class="status-dot"></span>
        Transcript loaded — ready to answer
    </div>
    """, unsafe_allow_html=True)

    if ask and query:
        retriever = vector_store.as_retriever()

        with st.spinner("Thinking…"):
            docs = retriever.invoke(query)
            context = " ".join([doc.page_content for doc in docs])

            lang_instruction = "Answer in Hindi." if language == "Hindi" else "Answer in English."

            final_prompt = f"""
Answer the question based only on the context provided. Be concise and clear.

{lang_instruction}

Context:
{context}

Question:
{query}
"""
            llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
            response = llm.invoke(final_prompt)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-tag">AI Answer</div>
            <div class="answer-text">{response.content}</div>
        </div>
        """, unsafe_allow_html=True)

    elif ask and not query:
        st.warning("Please type a question before asking.")


st.markdown("""
<div class="yt-footer">
    YT Insight · LangChain
</div>
""", unsafe_allow_html=True)