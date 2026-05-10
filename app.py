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

COOKIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
COOKIES_AVAILABLE = os.path.exists(COOKIES_PATH)

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


def is_rate_limit_error(error):
    if isinstance(error, HTTPError) and error.code == 429:
        return True
    err_str = str(error).lower()
    return "429" in err_str or "rate limit" in err_str or "too many requests" in err_str


def backoff_delay_seconds(attempt):
    return (2 ** attempt) + random.uniform(0, 0.5)


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
    return f"YouTube is rate limiting requests. Please wait about {remaining_minutes} minute(s) and try again."


def set_rate_limit_cooldown(video_id, cooldown_seconds=90):
    cooldowns = st.session_state.get("yt_rate_limit_cooldowns", {})
    cooldowns[video_id] = time.time() + cooldown_seconds
    st.session_state["yt_rate_limit_cooldowns"] = cooldowns


def remove_html_tags(text):
    result = ""
    inside_tag = False
    for ch in text:
        if ch == "<":
            inside_tag = True
        elif ch == ">":
            inside_tag = False
        elif not inside_tag:
            result += ch
    return result


def remove_curly_braces(text):
    parts = text.split("{")
    cleaned = parts[0]
    for part in parts[1:]:
        if "}" in part:
            cleaned += part.split("}", 1)[1]
        else:
            cleaned += part
    return cleaned


def is_timestamp_line(text):
    if len(text) >= 5 and text[0].isdigit() and text[1].isdigit() and text[2] == ":" and text[3].isdigit() and text[4].isdigit():
        return True
    return False


def load_transcript_via_api(video_id, max_attempts=3):
    last_error = None
    cookie_kwargs = {"cookies": COOKIES_PATH} if COOKIES_AVAILABLE else {}

    for attempt in range(max_attempts):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, **cookie_kwargs)

            transcript = None
            try:
                transcript = transcript_list.find_transcript(["en", "hi"])
            except Exception:
                try:
                    for lang in transcript_list._manually_created_transcripts:
                        transcript = transcript_list._manually_created_transcripts[lang]
                        break
                except Exception:
                    pass

                if transcript is None:
                    for candidate in transcript_list:
                        transcript = candidate
                        break

            if transcript is None:
                raise ValueError("No transcript track was found for this video.")

            transcript_chunks = transcript.fetch()
            text = " ".join(
                chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
                for chunk in transcript_chunks
            ).strip()

            if text:
                return text
            raise ValueError("Transcript text was empty after fetching captions.")

        except Exception as error:
            last_error = error
            if is_rate_limit_error(error) and attempt < max_attempts - 1:
                time.sleep(backoff_delay_seconds(attempt))
                continue
            raise

    raise last_error


def _ytdlp_extract(video_url, ydl_opts, max_attempts=3):
    last_error = None
    for attempt in range(max_attempts):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(video_url, download=False)
        except Exception as error:
            last_error = error
            if is_rate_limit_error(error) and attempt < max_attempts - 1:
                time.sleep(backoff_delay_seconds(attempt))
                continue
            break
    return None


def _pick_subtitle_url(subtitle_set):
    preferred = ["en", "en-US", "en-GB", "hi", "en-IN"]
    for lang in preferred:
        tracks = subtitle_set.get(lang)
        if tracks:
            return tracks[0].get("url", "")
    for lang in subtitle_set:
        if "en" in lang.lower():
            tracks = subtitle_set.get(lang)
            if tracks:
                return tracks[0].get("url", "")
    for lang in sorted(subtitle_set):
        tracks = subtitle_set.get(lang)
        if tracks:
            return tracks[0].get("url", "")
    return ""


def _subtitle_url_to_text(subtitle_url):
    req = urllib.request.Request(
        subtitle_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            break
        except Exception as error:
            if is_rate_limit_error(error) and attempt < 2:
                time.sleep(backoff_delay_seconds(attempt))
                continue
            raise

    cleaned = []
    for line in raw.splitlines():
        s = line.strip()

        if not s:
            continue
        if s in ("WEBVTT", "NOTE"):
            continue
        if s.isdigit():
            continue
        if "-->" in s:
            continue
        if is_timestamp_line(s):
            continue
        if s.startswith("STYLE") or s.startswith("Style:") or s.startswith("NOTE "):
            continue

        s = remove_html_tags(s)
        s = remove_curly_braces(s)
        s = html.unescape(s)
        s = " ".join(s.split())

        if s:
            cleaned.append(s)

    result = " ".join(cleaned).strip()
    if not result:
        raise ValueError("Subtitle track contained no extractable text.")
    return result


def load_transcript_via_ytdlp(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    def build_opts(extra=None):
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "hi", "en-US", "en-GB"],
            "socket_timeout": 30,
        }
        if COOKIES_AVAILABLE:
            opts["cookiefile"] = COOKIES_PATH
        if extra:
            opts.update(extra)
        return opts

    info = _ytdlp_extract(video_url, build_opts(), max_attempts=2)

    if info is None and not COOKIES_AVAILABLE:
        try:
            info = _ytdlp_extract(video_url, build_opts({"cookiesfrombrowser": ("chrome",)}), max_attempts=1)
        except Exception:
            pass

    if info is None and not COOKIES_AVAILABLE:
        try:
            info = _ytdlp_extract(video_url, build_opts({"cookiesfrombrowser": ("firefox",)}), max_attempts=1)
        except Exception:
            pass

    if info is None:
        raise ValueError("yt-dlp could not retrieve video information.")

    subtitle_sets = [info.get("subtitles") or {}, info.get("automatic_captions") or {}]
    for subtitle_set in subtitle_sets:
        subtitle_url = _pick_subtitle_url(subtitle_set)
        if subtitle_url:
            try:
                text = _subtitle_url_to_text(subtitle_url)
                if text.strip():
                    return text
            except Exception:
                continue

    raise ValueError("No usable subtitle track was found via yt-dlp.")


def load_transcript_text(video_id):
    if not video_id:
        raise ValueError("Please provide a valid YouTube URL.")

    cooldown_error = get_rate_limit_cooldown_error(video_id)
    if cooldown_error:
        raise ValueError(cooldown_error)

    primary_error = None
    try:
        return load_transcript_via_api(video_id)
    except Exception as err:
        primary_error = err

    fallback_error = None
    try:
        return load_transcript_via_ytdlp(video_id)
    except Exception as err:
        fallback_error = err

    if is_rate_limit_error(primary_error) or is_rate_limit_error(fallback_error):
        set_rate_limit_cooldown(video_id)
        raise ValueError(
            "YouTube is blocking requests from this app (HTTP 429).\n\n"
            "To fix this:\n"
            "1. Export cookies.txt from Chrome using 'Get cookies.txt LOCALLY' extension\n"
            "2. Place cookies.txt in the same folder as app.py\n"
            "3. Restart the app"
        )

    raise ValueError(
        "Could not load transcript. Please check:\n"
        "• The video URL is valid and the video is public\n"
        "• The video has captions or subtitles enabled\n"
        "• The video is not age-restricted\n\n"
        f"Details: {fallback_error or primary_error}"
    )


@st.cache_resource
def process_video(url):
    try:
        video_id = extract_video_id(url)
        transcript_text = load_transcript_text(video_id)
        if not transcript_text:
            return None, "Transcript text was empty for this video."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
        chunks = splitter.split_text(transcript_text)

        if not chunks:
            return None, "Transcript could not be split into searchable chunks."

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store, None
    except Exception as error:
        return None, str(error)


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
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})

        with st.spinner("Thinking…"):
            docs = retriever.invoke(query)
            context = " ".join([doc.page_content for doc in docs])
            lang_instruction = "Answer in Hindi." if language == "Hindi" else "Answer in English."

            final_prompt = f"""You are given a transcript from a YouTube video. Answer the question using only the information present in the transcript below. Do not say where something starts or give timestamps. Just explain what the transcript says about the topic directly.

{lang_instruction}

Transcript:
{context}

Question:
{query}"""

            llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
            response = llm.invoke(final_prompt)

        answer_html = response.content.replace("\n", "<br>")
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-tag">AI Answer</div>
            <div class="answer-text">{answer_html}</div>
        </div>
        """, unsafe_allow_html=True)

    elif ask and not query:
        st.warning("Please type a question before asking.")


st.markdown("""
<div class="yt-footer">
    YT Insight · LangChain
</div>
""", unsafe_allow_html=True)