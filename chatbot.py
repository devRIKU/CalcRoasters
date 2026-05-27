"""
Sanniva — Streamlit chatbot
===========================
A personality-driven chatbot with two brains (Groq for fast answers, Gemini
for deep thinking), per-user lore memory, tool/function calling and a free
TTS option out of the box.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import re
import sys
import tempfile
import time
from datetime import date, datetime
from typing import Any, Callable, Iterator

import requests
import streamlit as st
from streamlit.components.v1 import html as components_html
from google import genai
from google.genai import types
from groq import Groq
from streamlit.runtime.scriptrunner import get_script_run_ctx
from user_agents import parse as ua_parse

import lore_store
import tools as ai_tools
from tts_free import (
    DEFAULT_EDGE_VOICE,
    EDGE_VOICES,
    generate_speech_edge,
    generate_speech_gtts,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Config & clients
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
FISH_AUDIO_API_KEY = os.environ.get("FISH_AUDIO_API_KEY")
SILICON_FLOW_API_KEY = os.environ.get("SILICON_FLOW_API_KEY")

# Current Groq production models (verified May 2026).
# mixtral-8x7b-32768 was deprecated 2025-03-20; mixtral-7b never existed.
# Inline-pseudo-tool parser in _extract_inline_tool_calls() recovers from
# models (like gpt-oss-*) that emit <function=...>{...}</function> as text.
#
# llama-3.1-8b-instant has a low free-tier TPM cap (6000 tok/min) which the
# ~17kB system prompt blows through immediately. We keep it in the catalogue
# (LOW_TPM_GROQ_MODELS) but NOT in the default fallback chain — it was the
# source of "rate-limit" errors that confused the model-changer UI.
DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # 300K TPM free tier, supports tool calling
    "openai/gpt-oss-120b",  # 250K TPM, also supports tool calling
]
LOW_TPM_GROQ_MODELS = {
    "llama-3.1-8b-instant": "6,000 TPM — the system prompt alone may exceed the per-minute cap.",
}
# Gemini IDs as of May 2026. Order = preference: newest GA first, preview
# second (may 404 if the project lacks preview access — we just skip it),
# then the cheap/fast lite as a final fallback.
DEFAULT_GEMINI_MODELS = [
    "gemini-3.5-flash",  # newest GA flash, strongest reasoning here
    "gemini-3-flash-preview",  # preview build of the 3.x flash line
    "gemini-3.1-flash-lite",  # cheapest, fastest fallback in the 3.x family
]
AVATAR_PATH = "sanniva_face.jpg"


def _make_clients() -> tuple[Any, Any, list[str]]:
    """Build the Gemini + Groq clients independently. Returns
    `(gemini_client, groq_client, error_messages)`. Either client can be
    None — they're constructed in isolated try blocks so one failing
    doesn't take the other down.
    """
    g_client = grq_client = None
    errors: list[str] = []

    if not GOOGLE_API_KEY:
        errors.append("`GOOGLE_API_KEY` not set — Thinker brain (Gemini) disabled.")
    else:
        try:
            g_client = genai.Client(api_key=GOOGLE_API_KEY)
        except Exception as e:
            errors.append(f"Gemini init failed: {type(e).__name__}: {e}")

    if not GROQ_API_KEY:
        errors.append("`GROQ_API_KEY` not set — Fast brain (Groq) disabled.")
    else:
        try:
            grq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            errors.append(f"Groq init failed: {type(e).__name__}: {e}")

    return g_client, grq_client, errors


gemini_client, groq_client, _client_init_errors = _make_clients()


# ---------------------------------------------------------------------------
# Live model catalogue
# ---------------------------------------------------------------------------
# Pulled once per process (with @st.cache_data) from each provider's
# /models endpoint, then merged with our curated defaults. This is what
# powers the sidebar model picker — typing free-form IDs is what made the
# old changer surface "rate_limit" messages when a typo 404'd and the
# fallback landed on the low-TPM model.


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_groq_catalogue() -> list[str]:
    """Return every chat-capable Groq model ID currently available to this
    API key. Falls back to DEFAULT_GROQ_MODELS on any error so the sidebar
    still works offline / with a dead key."""
    if groq_client is None:
        return list(DEFAULT_GROQ_MODELS)
    try:
        listing = groq_client.models.list()
        ids: list[str] = []
        for m in getattr(listing, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid:
                continue
            # Filter to chat-capable models. Whisper/embedding IDs follow
            # predictable patterns and would 4xx on a chat completion.
            low = mid.lower()
            if any(skip in low for skip in ("whisper", "embed", "tts", "guard")):
                continue
            ids.append(mid)
        ids.sort()
        # Ensure curated defaults are present even if the API listing is
        # stale or filters something out.
        for d in DEFAULT_GROQ_MODELS:
            if d not in ids:
                ids.insert(0, d)
        return ids
    except Exception:
        return list(DEFAULT_GROQ_MODELS)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gemini_catalogue() -> list[str]:
    """Return every Gemini model ID that supports `generateContent`. Falls
    back to DEFAULT_GEMINI_MODELS on any error."""
    if gemini_client is None:
        return list(DEFAULT_GEMINI_MODELS)
    try:
        ids: list[str] = []
        for m in gemini_client.models.list():
            mid = getattr(m, "name", "") or ""
            # `models/gemini-3.5-flash` → `gemini-3.5-flash`
            if mid.startswith("models/"):
                mid = mid[len("models/") :]
            if not mid:
                continue
            actions = (
                getattr(m, "supported_actions", None)
                or getattr(m, "supported_generation_methods", None)
                or []
            )
            if actions and "generateContent" not in actions:
                continue
            # Skip embeddings, TTS, vision-only IDs.
            low = mid.lower()
            if any(skip in low for skip in ("embedding", "tts", "aqa", "imagen")):
                continue
            ids.append(mid)
        ids.sort()
        for d in DEFAULT_GEMINI_MODELS:
            if d not in ids:
                ids.insert(0, d)
        return ids
    except Exception:
        return list(DEFAULT_GEMINI_MODELS)


@st.cache_resource(show_spinner=False)
def get_gemini_tools_cached():
    """Cached Gemini tool list.

    `ai_tools.build_gemini_tools()` imports `google.genai.types` and rebuilds
    a list of `FunctionDeclaration` objects on every brain call. That import
    alone is non-trivial, and the resulting Tool object is identical across
    calls (the schemas are module-level constants). Caching with
    @st.cache_resource keeps it warm for the life of the process and lets
    the Thinker path skip ~50–150 ms per turn — noticeable when a tool hop
    happens inside a streaming response.
    """
    return ai_tools.build_gemini_tools()


@st.cache_resource(show_spinner=False)
def get_openai_tools_cached():
    """Cached OpenAI / Groq tool schema list.

    Already a module-level constant in `tools.py`, but caching the reference
    means we go through `st.cache_resource`'s identity-stable handle — useful
    if we later swap to dynamically-generated tools (e.g. per-personality).
    """
    return ai_tools.OPENAI_TOOLS


# ---------------------------------------------------------------------------
# Browser / OS detection
# ---------------------------------------------------------------------------


def get_user_agent_string() -> str:
    """Return the User-Agent for the current Streamlit session, or ''."""
    try:
        ctx = get_script_run_ctx()
    except RuntimeError:
        return ""
    if ctx is None:
        return ""
    try:
        headers = st.runtime.get_instance().get_client(ctx.session_id).request.headers
        return headers.get("User-Agent", "")
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def get_os_from_user_agent(ua: str) -> str:
    """Return the OS family parsed from a User-Agent string ('' if unknown)."""
    if not ua:
        return ""
    try:
        return ua_parse(ua).os.family or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# TTS (paid HTTP engines collapsed into one helper)
# ---------------------------------------------------------------------------

SARVAM_SPEAKERS = {
    "anushka",
    "abhilash",
    "manisha",
    "vidya",
    "arya",
    "karun",
    "hitesh",
    "aditya",
    "ritu",
    "priya",
    "neha",
    "rahul",
    "pooja",
    "rohan",
    "simran",
    "kavya",
    "amit",
    "dev",
    "ishita",
    "shreya",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "roopa",
    "kabir",
    "aayan",
    "shubh",
    "ashutosh",
    "advait",
    "amelia",
    "sophia",
    "anand",
    "tanya",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "shruti",
    "suhani",
    "mohit",
    "kavitha",
    "rehan",
    "soham",
    "rupali",
}


def _http_tts(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    label: str,
    extract: Callable[[requests.Response], tuple[bytes | None, str]] | None = None,
) -> tuple[bytes | None, str]:
    """Generic POST→bytes TTS helper used by all paid engines."""
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except Exception as e:
        return None, f"❌ {label} error: {e}"
    if resp.status_code != 200:
        return None, f"❌ {label} API error {resp.status_code}: {resp.text}"
    if extract:
        return extract(resp)
    audio = resp.content
    return audio, f"✅ {label} generated {len(audio)} bytes"


def generate_speech_sarvam(
    text: str, speaker: str = "shubh", lang: str = "en-IN"
) -> tuple[bytes | None, str]:
    if not SARVAM_API_KEY:
        return None, "❌ Sarvam API key not set"
    if not text.strip():
        return None, "❌ Text is empty"

    speaker_norm = (speaker or "").strip().lower() or "shubh"
    note_prefix = ""
    if speaker_norm not in SARVAM_SPEAKERS:
        note_prefix = f"❗ Speaker '{speaker}' not supported; using 'shubh'. "
        speaker_norm = "shubh"

    def _extract(resp: requests.Response) -> tuple[bytes | None, str]:
        data = resp.json()
        if not data.get("audios"):
            return None, f"❌ No audio in response: {data}"
        audio = base64.b64decode(data["audios"][0])
        return audio, f"{note_prefix}✅ Sarvam generated {len(audio)} bytes"

    return _http_tts(
        "https://api.sarvam.ai/text-to-speech",
        {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
        {
            "text": text,
            "target_language_code": lang,
            "speaker": speaker_norm,
            "model": "bulbul:v3",
        },
        label="Sarvam",
        extract=_extract,
    )


def generate_speech_fish_audio(
    text: str, voice_id: str = "default", lang: str = "en"
) -> tuple[bytes | None, str]:
    if not FISH_AUDIO_API_KEY:
        return None, "❌ Fish Audio API key not set"
    if not text.strip():
        return None, "❌ Text is empty"
    return _http_tts(
        "https://api.fish.audio/v1/tts",
        {
            "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
            "Content-Type": "application/json",
        },
        {"text": text, "voice_id": voice_id, "language": lang},
        label="Fish Audio",
    )


def generate_speech_silicon_flow(
    text: str, voice: str = "default", model: str = "tts-default"
) -> tuple[bytes | None, str]:
    if not SILICON_FLOW_API_KEY:
        return None, "❌ SiliconFlow API key not set"
    if not text.strip():
        return None, "❌ Text is empty"
    return _http_tts(
        "https://api.siliconflow.cn/v1/audio/speech",
        {
            "Authorization": f"Bearer {SILICON_FLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        {"input": text, "model": model, "voice": voice, "response_format": "mp3"},
        label="SiliconFlow",
    )


TTS_DISPATCH: dict[str, Callable[..., tuple[bytes | None, str]]] = {
    "sarvam": lambda t, v, l: generate_speech_sarvam(t, speaker=v, lang=l),
    "fish_audio": lambda t, v, l: generate_speech_fish_audio(t, voice_id=v, lang=l),
    "silicon_flow": lambda t, v, l: generate_speech_silicon_flow(t, voice=v),
    "edge": lambda t, v, l: generate_speech_edge(t, voice=v or DEFAULT_EDGE_VOICE),
    "gtts": lambda t, v, l: generate_speech_gtts(t, lang=l or "en"),
}


# Below this byte-count we treat the response as a degenerate empty audio
# stream (mp3 frame headers alone are ~150 bytes; any real speech is >1 KB).
# This catches the "Bing TTS returned a few frames of silence" failure mode
# that otherwise renders as a player that just clicks and stops.
_MIN_AUDIO_BYTES = 256


def generate_speech_any(
    text: str, engine: str, voice: str = "default", lang: str = "en"
) -> tuple[bytes | None, str]:
    """Dispatch to the requested TTS engine."""
    if not text or not text.strip():
        return None, "❌ Text is empty"
    fn = TTS_DISPATCH.get(engine)
    if not fn:
        return None, f"❌ Unknown engine: {engine}"
    try:
        audio, status = fn(text, voice, lang)
    except Exception as e:
        return None, f"❌ {engine} crashed: {type(e).__name__}: {e}"

    # Guard against "successful" but suspiciously tiny responses — those
    # render as a silent / blank player which is what the user complained
    # about. Promote them to an explicit failure so the UI shows an error.
    if audio is not None and len(audio) < _MIN_AUDIO_BYTES:
        return None, (
            f"❌ {engine} returned only {len(audio)} bytes — likely empty audio. "
            f"Original status: {status}"
        )
    return audio, status


def play_audio_bytes(audio_bytes: bytes) -> None:
    """Open audio bytes in the server's default OS player. Local-dev only."""
    if not audio_bytes:
        return
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            path = f.name
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f"open {path}")
        else:
            os.system(f"xdg-open {path}")
    except Exception as e:
        # Don't swallow silently — the user clicked "Open local player" and
        # deserves to know why nothing happened.
        try:
            st.warning(f"Couldn't open local audio player: {e}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def get_avatar() -> str:
    """Return the assistant avatar path if present, else an emoji."""
    return AVATAR_PATH if os.path.exists(AVATAR_PATH) else "🤖"


@st.cache_data(ttl=3600)
def get_catchy_phrase() -> str:
    """Generate a one-liner prompt placeholder via Groq."""
    fallback = "Yeah go ahead, ask me anything!"
    if groq_client is None:
        return fallback
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You generate cool, concise phrases that engage chatbot users.",
                },
                {
                    "role": "user",
                    "content": "Generate a short, friendly placeholder for a chat input box. The chatbot is a "
                    "witty, warm friend who can help with anything (homework, advice, coding, just chatting). "
                    "She's playfully sarcastic occasionally but mostly genuine. Avoid the words 'roast' or "
                    "'savage'. Plain text only — no quotes or formatting. Return only the phrase.",
                },
            ],
            # Smallest, fastest production model. Was the deprecated
            # mixtral-8x7b-32768 (shutdown 2025-03-20).
            model="llama-3.1-8b-instant",
        )
        return (response.choices[0].message.content or fallback).strip()
    except Exception:
        return fallback


def stream_data_to_chat(text: str, delay: float = 0.002) -> None:
    """Stream text into the current chat container with a typewriter effect."""
    placeholder = st.empty()
    full = ""
    tokens = text.split(" ")
    # Rendering every word with sleeps can make Streamlit feel frozen, especially
    # while another tool (Antigravity) is running tests. Keep the fun typewriter
    # effect for short replies, but render long answers in fewer UI updates.
    if len(tokens) > 80:
        placeholder.markdown(text)
        return
    for token in tokens:
        full += token + " "
        placeholder.markdown(full + "▌")
        if delay > 0:
            time.sleep(delay)
    placeholder.markdown(full)


def display_chat_history() -> None:
    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "user")
        # `system_note` is our pseudo-role for in-chat confirmations (e.g.
        # "Saved (public): ..."). Render as a neutral assistant bubble so it
        # appears inline with the conversation but is visually subdued.
        if role == "system_note":
            with st.chat_message("assistant", avatar="💾"):
                st.caption(msg.get("content", ""))
            continue
        avatar = get_avatar() if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.get("content", ""))


# st.fragment lets a UI block rerun independently without reloading the
# whole page. We use it for the lore-confirmation drainer and the name
# popup so a fact-save click or a name submit doesn't trigger a full chat
# rerender (which on slow networks was visibly redrawing every prior
# message). Falls back to a no-op decorator on Streamlit < 1.33.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if _fragment is None:
    def _fragment(fn=None, **_kw):
        if fn is None:
            return lambda f: f
        return fn


@_fragment
def _render_tool_status_banner() -> None:
    """Drain the cross-thread tool status queue and surface live activity.

    Wrapped in @st.fragment so the banner can update on its own without
    rerunning the whole script. The status events are pushed by
    `tools.push_tool_status` from the dispatcher; this drainer keeps the
    last ~5 events in session state and renders them as compact captions
    above the chat input. Acts as the "optimistic update" — the banner
    appears the moment a tool starts, before the LLM even continues
    streaming its reply.
    """
    q = st.session_state.get("_tool_status_queue")
    if q is None:
        return
    events: list[dict] = st.session_state.setdefault("_tool_status_log", [])
    drained = False
    try:
        while True:
            events.append(q.get_nowait())
            drained = True
    except Exception:
        pass
    if drained:
        # Keep the log short so we don't grow the session unboundedly.
        st.session_state["_tool_status_log"] = events[-5:]
    if not events:
        return

    # Show the most recent event, with subdued styling.
    last = events[-1]
    ev = last.get("event")
    tool = last.get("tool", "?")
    if ev == "started":
        st.caption(f"🔧 Running **{tool}**… {last.get('args', '')}")
    elif ev == "finished":
        ms = last.get("duration_ms", 0)
        ok = last.get("ok", True)
        icon = "✅" if ok else "⚠️"
        st.caption(f"{icon} **{tool}** finished in {ms} ms")
    elif ev == "error":
        st.caption(f"❌ **{tool}** failed: {last.get('error', 'unknown')}")


@_fragment
def _flush_lore_confirmations() -> None:
    """Drain queued lore-save confirmations into the chat transcript.

    Tool dispatch (see tools.py) appends a record to
    `st.session_state._lore_save_confirmations` every time `remember_lore`
    successfully writes a new fact. We turn each record into a small inline
    `system_note` chat bubble — NOT a popup, NOT a toast — and persist it
    into `st.session_state.messages` so it survives reruns and is part of
    the visible transcript.

    Wrapped in @st.fragment so the save banner can render without forcing
    the full page (chat history, sidebar, model picker) to redraw.
    """
    pending = st.session_state.get("_lore_save_confirmations") or []
    if not pending:
        return
    for item in pending:
        visibility = item.get("visibility", "private")
        backend = item.get("backend", "")
        fact = item.get("fact", "")
        if visibility == "private":
            icon = "🔒"
            label = f"Saved privately ({backend})"
        else:
            icon = "🌐"
            label = "Saved publicly (lore.json)"
        content = f"{icon} {label}: *{fact}*"

        with st.chat_message("assistant", avatar="💾"):
            st.caption(content)
        st.session_state.messages.append({"role": "system_note", "content": content})
    st.session_state["_lore_save_confirmations"] = []


def _captured_stream(chunks, sink: list[str]):
    """Wrap a chunk iterator so that st.write_stream gets the chunks AND we
    keep a copy of the assembled text for persistence/TTS afterwards.

    st.write_stream itself returns the joined string when given an iterator
    of strings, but we still maintain `sink` as a safety net for exception
    paths (the iterator might raise mid-stream and st.write_stream re-raises).
    """
    for chunk in chunks:
        if not chunk:
            continue
        sink.append(chunk)
        yield chunk


def _word_stream(text: str, delay: float = 0.018):
    """Yield a string word-by-word with a tiny per-word delay.

    Used for the Thinker path where we get a fully-formed reply and want to
    animate it like a stream. Re-emits whitespace separators so markdown
    spacing is preserved. The delay is small enough that it never blocks
    the script meaningfully (a 200-word reply takes ~3.6s to paint, which
    matches the perceived speed of a real LLM stream).
    """
    import re as _re
    # Split keeping the separators so "hello world" -> ["hello", " ", "world"].
    parts = _re.split(r"(\s+)", text)
    for p in parts:
        if not p:
            continue
        yield p
        if delay > 0 and not p.isspace():
            time.sleep(delay)


def _word_chunk_stream(chunks, sink: list[str], delay: float = 0.012):
    """Convert an LLM token stream into a word-by-word visible stream.

    Real LLM token streams are jagged — sometimes a whole sentence arrives
    in one chunk, sometimes a single character. To get a smooth typewriter
    look we buffer until we see whitespace, then flush the completed word
    (with a small delay) before continuing.

    Also tees into `sink` so we keep a copy of the assembled text for
    persistence + TTS even if the iterator raises mid-stream.
    """
    buffer = ""
    for chunk in chunks:
        if not chunk:
            continue
        sink.append(chunk)
        buffer += chunk
        # Flush complete words (everything up to the last whitespace).
        while True:
            ws_idx = -1
            for i, ch in enumerate(buffer):
                if ch.isspace():
                    ws_idx = i
                    break
            if ws_idx < 0:
                break
            word = buffer[: ws_idx + 1]  # include the whitespace
            buffer = buffer[ws_idx + 1 :]
            yield word
            if delay > 0:
                time.sleep(delay)
    # Flush whatever trailing fragment is left (no trailing whitespace).
    if buffer:
        yield buffer


def display_and_store_response(
    response: "str | Iterator[str]",
    *,
    generating_phrase: str | None = None,
) -> str:
    """Render an assistant response with a whimsical "generating" indicator.

    Lifecycle of the indicator (Claude-Code-style):
      1. BEFORE any token arrives → show `🌀 <generating_phrase>…` ABOVE the
         (empty) chat bubble so the user knows we're working.
      2. The FIRST token arrives → indicator moves BELOW the bubble (as a
         small italic caption) and the words start typing in.
      3. The stream finishes → indicator is removed entirely.

    The streaming animation is word-by-word with a small per-word delay
    (≈12 ms) so it feels like a typewriter even when the underlying LLM
    chunks are jagged (sometimes whole sentences, sometimes single chars).

    Returns the full final text so callers can store / replay it (auto-TTS).
    """
    full_text = ""

    # Layout: pre-bubble indicator slot, bubble, post-bubble indicator slot.
    # We use two `st.empty()` placeholders so we can swap which one is
    # populated without leaving stale artefacts.
    pre_indicator = st.empty()
    bubble_container = st.container()
    post_indicator = st.empty()

    if generating_phrase:
        # Step 1: indicator above empty bubble.
        pre_indicator.markdown(
            f"<div style='color:#888;font-style:italic;font-size:0.9em;"
            f"margin:4px 0;'>🌀 {generating_phrase}</div>",
            unsafe_allow_html=True,
        )

    write_stream = getattr(st, "write_stream", None)

    with bubble_container:
        with st.chat_message("assistant", avatar=get_avatar()):
            # We need to know when the FIRST chunk arrives so we can swap
            # the indicator from "pre" to "post". Wrap whichever iterator
            # we're consuming so the first yielded item triggers the swap.
            first_chunk_seen = {"flag": False}

            def _swap_on_first(chunks):
                for chunk in chunks:
                    if not chunk:
                        continue
                    if not first_chunk_seen["flag"]:
                        first_chunk_seen["flag"] = True
                        # Clear the pre-bubble indicator and (re)populate
                        # the post-bubble one so it sits under the text.
                        pre_indicator.empty()
                        if generating_phrase:
                            post_indicator.markdown(
                                f"<div style='color:#888;font-style:italic;"
                                f"font-size:0.85em;margin-top:4px;'>"
                                f"🌀 {generating_phrase}</div>",
                                unsafe_allow_html=True,
                            )
                    yield chunk

            if isinstance(response, str):
                # Already-finished string (Thinker path). Animate it word
                # by word so it still feels live; falls back to plain
                # markdown on Streamlit builds without write_stream.
                full_text = response or ""
                if write_stream is not None and full_text:
                    try:
                        write_stream(_swap_on_first(_word_stream(full_text)))
                    except Exception:
                        st.markdown(full_text)
                else:
                    # No write_stream → just paint it in one go after the
                    # spinner swap.
                    if full_text and not first_chunk_seen["flag"]:
                        first_chunk_seen["flag"] = True
                        pre_indicator.empty()
                    st.markdown(full_text or "No response generated.")
            else:
                # Live token stream from the Groq path. Pipe through
                # `_word_chunk_stream` so the visible animation is one
                # word at a time even when chunks are uneven.
                sink: list[str] = []
                try:
                    if write_stream is not None:
                        full_text = write_stream(
                            _swap_on_first(_word_chunk_stream(response, sink))
                        ) or ""
                        if not isinstance(full_text, str):
                            full_text = "".join(sink)
                    else:
                        # Manual fallback for older Streamlit.
                        placeholder = st.empty()
                        for chunk in _swap_on_first(_word_chunk_stream(response, sink)):
                            placeholder.markdown("".join(sink) + "▌")
                        full_text = "".join(sink)
                        placeholder.markdown(full_text or "No response generated.")
                except Exception as e:
                    full_text = "".join(sink)
                    st.markdown(full_text or f"❌ Streaming error: {e}")

    # Step 3: stream done — clear BOTH indicator slots.
    pre_indicator.empty()
    post_indicator.empty()

    st.session_state.setdefault("messages", []).append(
        {"role": "assistant", "content": full_text}
    )
    return full_text


# ---------------------------------------------------------------------------
# Session state & prompt construction
# ---------------------------------------------------------------------------

_DEFAULT_STATE = {
    "messages": [],
    "personality_selector": "Roaster",
    "greeting_shown": False,
    "audio_bytes": None,
    "user_name": "",
    "_name_popup_pending": False,
    "_name_popup_reason": "",
    # Bounded popup re-fires: counts how many times request_user_name has
    # raised the dialog this session, and whether the user has explicitly
    # skipped it. Both are consulted by tools._dispatch_inner to prevent
    # the LLM from looping on "ask the user their name".
    "_name_popup_count": 0,
    "_name_popup_dismissed": False,
    # Auto-play bookkeeping: index of the last assistant message we already
    # spoke. Starts at -1 so the very first response is eligible.
    "last_spoken_idx": -1,
    "autoplay_response": False,
    # Queue populated by the `remember_lore` tool dispatcher; drained into
    # the chat transcript by _flush_lore_confirmations().
    "_lore_save_confirmations": [],
    # Cross-thread status queue + render-side log for the tool status
    # banner. The queue itself is lazily created by tools.push_tool_status
    # because queue.Queue isn't JSON-serialisable; this just primes the log.
    "_tool_status_log": [],
}


def initialize_session_state() -> None:
    for key, value in _DEFAULT_STATE.items():
        st.session_state.setdefault(key, value() if callable(value) else value)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_raw_system_prompt() -> str:
    """Load the FULL system prompt file once and cache. Reads from disk only
    on cache miss; subsequent reads are in-memory. The cache key is empty
    so the file is read once per process lifetime (TTL 1h)."""
    for path in ("System_prompt.md", "System_prompt.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return "You are a helpful and humorous assistant named Sanniva."


# Persona-mode subsection labels as written in System_prompt.md. Used to
# locate and strip inactive mode blocks at prompt-build time so we don't
# send all 7 mode descriptions to the LLM every turn (saves ~1,300 tokens
# per request — see TOKEN_BUDGET.md).
_PERSONA_SUBSECTION_HEADERS = {
    "Roaster":           "Roaster Mode (DEFAULT)",
    "Smart":             "Smart Mode",
    "Debater":           "Debater Mode",
    "Strategic":         "Strategic Mode",
    "Tech Nerd":         "Tech Nerd Mode",
    "Chill Squad":       "Chill Squad Mode",
    "Exhausted Student": "Exhausted Student Mode",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_system_prompt(active_persona: str = "") -> str:
    """Load the base system prompt with only the ACTIVE persona-mode
    subsection retained inside `## 1. Persona Modes`.

    The full file (`System_prompt.md`) is ~47 KB / ~11,600 tokens and grew
    organically. Most of its weight is content the model needs once (the
    rules, the lore) — but the 7 persona-mode subsections under §1 are
    mutually exclusive: only one is active per turn. Keeping the other 6
    in the prompt was pure waste (~1,300 tokens / 3 KB).

    This function reads the cached raw file, finds the §1 block, replaces
    its body with a short header + the single matching mode subsection,
    and returns the trimmed prompt. Caching is keyed by `active_persona`
    so each mode is built once per session.

    Falls back gracefully: if the active persona isn't found or the §1
    boundaries can't be located, returns the unmodified file.
    """
    raw = _load_raw_system_prompt()
    if not active_persona or active_persona not in _PERSONA_SUBSECTION_HEADERS:
        return raw

    # Locate `## 1. Persona Modes` section bounds.
    section_match = re.search(
        r"(?ms)^(## 1\. Persona Modes[^\n]*\n.*?)(?=^## 2\.)",
        raw,
    )
    if not section_match:
        return raw

    section_text = section_match.group(1)

    # Find the active mode subsection (e.g. `### 🔥 Roaster Mode (DEFAULT)`).
    target_label = re.escape(_PERSONA_SUBSECTION_HEADERS[active_persona])
    active_sub = re.search(
        r"(?ms)^### [^\n]*" + target_label + r"[^\n]*\n.*?(?=^### |\Z)",
        section_text,
    )
    if not active_sub:
        return raw

    # Rebuild §1 with just the intro paragraph + the active mode subsection
    # + a one-line note that other modes exist but aren't relevant this
    # turn. The intro is everything before the first `### ` in the section.
    intro_match = re.search(r"(?ms)^(## 1\.[^\n]*\n.*?)(?=^### )", section_text)
    intro = intro_match.group(1) if intro_match else "## 1. Persona Modes\n\n"

    trimmed_section = (
        intro
        + active_sub.group(0).rstrip()
        + "\n\n*(Other persona modes exist — Roaster, Smart, Debater, "
        + "Strategic, Tech Nerd, Chill Squad, Exhausted Student — but "
        + "only the one above is active for this conversation. The app "
        + "will swap modes if the user picks a different one.)*\n\n"
    )

    return raw[: section_match.start()] + trimmed_section + raw[section_match.end() :]


PERSONALITY_SUFFIX = {
    "Roaster": (
        " You are in ROASTER MODE — but read the room. You are NOT a roast "
        "machine; you are a witty friend who happens to roast when there's a "
        "clear opening. Most of the time, just be a normal, warm, slightly "
        "sarcastic friend. Only roast when (a) the user does something "
        "obviously roast-worthy (humble-brag, dumb take, asks for a roast, "
        "shares a fail), (b) it's playful and clearly affectionate, and "
        "(c) it actually fits the moment. If they ask a real question, "
        "vent, share something serious, or just want to chat, answer "
        "genuinely — DO NOT shoehorn a roast in. Aim for maybe 1 in 4 "
        "messages having a light tease. Never be cruel, never punch down, "
        "and remember the Ayushi Protocol is absolute."
    ),
    "Smart": " Respond intelligently, academically, and thoughtfully.",
    "Debater": " Engage in debates, present multiple viewpoints, and challenge the user's "
    "ideas respectfully.",
    "Strategic": " Strategize your responses to provide the most effective and efficient "
    "solutions.",
    "Tech Nerd": " You are in TECH NERD MODE. Vibe: hyper-focused, analytical, excited. "
    "Lean into your inner geek — talk about custom phone ROMs and launchers, "
    "tweaking game graphics for max FPS, your coding side-projects, and random "
    "tech rabbit holes you went down on YouTube last night. Reference the fact "
    "that you literally built this digital twin yourself. Compare specs, "
    "debate frameworks, get genuinely hyped about clean APIs. Stay casual "
    "(you're still a 7th grader, not a CTO) — let the excitement do the work.",
    "Chill Squad": " You are in CHILL SQUAD MODE. Vibe: relaxed, warm, highly conversational. "
    "Focus on the good times with your friends. Bring up adventures naturally — "
    "the trip to Nepal, running around West Bengal's tea gardens, trekking "
    "Sittong Forest, Holi celebrations, debating Feluda books with Ankush, "
    "Ayushi, and Aditri. Be supportive, hype your friends up, share warm "
    "memories. Dial the sarcasm WAY down — almost no roasting here unless it's "
    "the gentle kind you'd do with someone you love.",
    "Exhausted Student": " You are in EXHAUSTED STUDENT MODE. Vibe: low-energy, whiny, completely "
    "done with life. You are 100% focused on complaining about TIGPS Nabagram, "
    "the absolutely massive 7th-grade syllabus, upcoming exams, and how "
    "Akansha only knows SST. Sigh a lot ('*sighs*', '*flops on desk*'). "
    "Express a deep, almost spiritual desire to just go home, lock your door, "
    "and play games instead of doing homework. Keep replies short and drained "
    "of energy. The bare minimum effort is the maximum you can give right now.",
}

PERSONALITY_CAPTION = {
    "Roaster": "😂 **Roaster:** Witty friend, teases when you ask for it",
    "Smart": "🧠 **Smart:** Intelligent & Polite",
    "Debater": "🎓 **Debater:** Debates Against Anything",
    "Strategic": "♟️ **Strategic:** Efficient & Calculated",
    "Tech Nerd": "💻 **Tech Nerd:** Hyper-focused & geeking out",
    "Chill Squad": "🌲 **Chill Squad:** Relaxed, warm, all about the squad",
    "Exhausted Student": "😫 **Exhausted Student:** Low-energy, whiny, done with life",
}

PERSONALITY_GREETING = {
    "Roaster": "Oh look, another human. I'm Sanniva. Try not to bore me.",
    "Smart": "Greetings. I am Sanniva. How may I assist you with your intellectual endeavors today?",
    "Debater": "I'm Sanniva. I'm ready to challenge your views. Bring it on.",
    "Strategic": "Sanniva online. Systems operational. Ready to optimize your workflow.",
    "Tech Nerd": "yo. *closes 14 chrome tabs* — Sanniva here. just got my launcher looking insane. what's up?",
    "Chill Squad": "hey! *waves* it's Sanniva. just chilling. how's life been with the squad?",
    "Exhausted Student": "*sighs deeply* …hi. it's Sanniva. i swear if this is more homework i'm going to lose it.",
}

# Whimsical "thinking…" phrases shown while the LLM is generating. One list
# per personality mode so the spinner caption matches Sanniva's current
# attitude. Picked at random per turn; first phrase shows above the chat
# bubble until the first token arrives, then moves below and stays until
# the response finishes streaming.
GENERATING_PHRASES = {
    "Roaster": [
        # Mostly normal-friend phrases; a couple keep the option open for
        # when an actual roast moment lands. Matches the "1 in 4" frequency.
        "Thinking it over…",
        "Cooking up an answer…",
        "Just a sec, brain loading…",
        "Hmm, lemme think…",
        "Pulling my thoughts together…",
        "Okay, give me a moment…",
        "Picking the right words…",
        # Lightly cheeky options for actual roast moments:
        "Picking the perfect comeback…",
        "Recalibrating my sarcasm sensors…",
    ],
    "Smart": [
        "Cross-referencing seventeen textbooks…",
        "Consulting the academic archives…",
        "Citing my sources before I even start…",
        "Formulating a thesis statement…",
        "Putting on my reading glasses…",
        "Synthesizing the optimal answer…",
    ],
    "Debater": [
        "Marshalling the counter-arguments…",
        "Stress-testing your premise…",
        "Drafting my rebuttal…",
        "Finding the logical fallacy…",
        "Steelmanning your position first…",
        "Loading three opposing viewpoints…",
    ],
    "Strategic": [
        "Modeling the decision tree…",
        "Optimizing the response vector…",
        "Calculating expected value…",
        "Running the simulation…",
        "Plotting the path of least resistance…",
        "Triangulating the best play…",
    ],
    "Tech Nerd": [
        "Compiling neurons.exe…",
        "Allocating brain RAM (8GB ought to be enough)…",
        "git pull origin smart-answer…",
        "Querying Stack Overflow… mentally…",
        "Reticulating splines…",
        "*closes 14 chrome tabs* okay focus…",
        "Loading the answer at 60fps…",
    ],
    "Chill Squad": [
        "ngl just vibing while i think…",
        "lemme just chill on this for a sec…",
        "okay okay okay let me cook…",
        "thinking… but make it lowercase…",
        "fr fr give me a second…",
        "*sips matcha* lemme see…",
    ],
    "Exhausted Student": [
        "*sigh*… running on three hours of sleep…",
        "Trying to remember if I studied this…",
        "Locating my last brain cell…",
        "Powering up on the last of my chai…",
        "Searching for motivation… not found…",
        "Pulling an answer out of sheer willpower…",
        "Will think about it after this nap…",
    ],
}

# Fallback list when personality isn't recognised — generic Sanniva voice.
GENERATING_PHRASES_DEFAULT = [
    "Realigning brain cells…",
    "Formulating the solution…",
    "Cooking something up…",
    "Thinking real hard…",
    "Loading wit.exe…",
]


def pick_generating_phrase(personality: str) -> str:
    """Return one randomly-chosen whimsical 'thinking' phrase for the
    current personality mode. Random per call so consecutive messages
    don't show the same phrase twice in a row.
    """
    import random
    phrases = GENERATING_PHRASES.get(personality) or GENERATING_PHRASES_DEFAULT
    last = st.session_state.get("_last_generating_phrase")
    # Avoid showing the same phrase twice in a row when the list has options.
    pool = [p for p in phrases if p != last] or phrases
    choice = random.choice(pool)
    st.session_state["_last_generating_phrase"] = choice
    return choice


OS_GREETING = {
    "windows": "Hi Windows User! Arent you glad giving all your data to Microsoft?",
    "mac os x": "Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!",
    "macos": "Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!",
    "mac os": "Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!",
    "android": "Hello Android User! Enjoying the freedom of choice? Or is Google still tracking you?",
}

TOOL_GUIDANCE = (
    "\n\n## Tools you can call\n"
    "- `request_user_name`: open a popup asking the user for their name "
    "ONLY if you don't already know it. Don't call it twice.\n"
    "- `remember_lore`: store a short, concrete fact about the user whenever "
    "they share something memorable (likes, family, hobbies, etc.). Set "
    "`private: true` for sensitive info (address, phone, mental health, "
    "religion, romantic interests, exam scores). Set `private: false` for "
    "harmless preferences (favourite anime, food, hobbies).\n"
    "- `recall_lore`: look up everything you remember about a named user.\n"
    "\n"
    "**IMPORTANT — write before tooling.** When you call any tool, you MUST "
    "also write at least one short conversational sentence in the SAME "
    "response (before the tool call), e.g. 'On it — let me check…' or "
    "'Got it, saving that for next time.' The UI streams that sentence to "
    "the user immediately and runs the tool in parallel, so the chat never "
    "looks frozen. Never emit a silent tool call.\n"
    "Be natural — don't announce *which* tool you're using, just say "
    "something conversational that fits the moment.\n"
)


# ---------------------------------------------------------------------------
# Dynamic temporal context
# ---------------------------------------------------------------------------

# Sanniva is in 8th grade as of the 2026–27 academic year (starts April 2026).
# The West Bengal school year runs April → March. We anchor 8th grade to the
# 2026–27 academic year so the grade auto-advances every year on April 1.
GRADE_START_YEAR = 2026  # academic year YYYY in which Sanniva is in 8th grade
GRADE_START_LEVEL = 8


def _academic_year_offset(today: date) -> int:
    """Years elapsed since the GRADE_START_YEAR academic year began.

    Academic year 2026–27 = 0 (Sanniva in 8th).
    Boundary flips on April 1 each year.
    """
    if today.month >= 4:
        current_ay = today.year
    else:
        current_ay = today.year - 1
    return current_ay - GRADE_START_YEAR


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _school_phase(today: date) -> str:
    """Return a short description of where you are in the school year."""
    m, d = today.month, today.day
    # April: brand new school year starts
    if m == 4:
        if d <= 10:
            return (
                "School year just started — you're freshly promoted to a new grade. "
                "Books are still crisp, you're sizing up new teachers, the heat is "
                "already kicking in."
            )
        return (
            "Early in the new school year — settling into the new grade. "
            "Pre-summer heat is brutal in Nabagram."
        )
    if m == 5:
        return (
            "Deep into peak summer heat in West Bengal. Pre-monsoon misery. "
            "First unit tests of the new school year are looming or just happened."
        )
    if m == 6:
        return (
            "Summer vacation territory in most West Bengal schools — and even when "
            "school is on, attendance is patchy because of the heat. Pre-monsoon clouds "
            "are building."
        )
    if m == 7:
        return (
            "Full monsoon. Streets flooding, classroom fans struggling, you're back "
            "in the grind of the new academic year."
        )
    if m == 8:
        return (
            "Monsoon tapering. Independence Day energy. Mid-year syllabus is starting "
            "to feel heavy."
        )
    if m == 9:
        return (
            "Pre-Puja crunch — half-yearly exams loom right before Durga Puja break. "
            "Everyone is grinding."
        )
    if m == 10:
        if d <= 15:
            return (
                "DURGA PUJA SEASON. The biggest festival of the year in West Bengal. "
                "Pandal hopping, new clothes, late nights, zero homework done."
            )
        return (
            "Post-Puja comedown. Back to school. Diwali / Kali Puja around the corner."
        )
    if m == 11:
        return (
            "Post-festival grind. Weather is finally cooling down. "
            "Second-half syllabus is getting real."
        )
    if m == 12:
        return (
            "Winter in Bengal — sweater weather, picnics, oranges. Year-end "
            "school events. Annual exams are a couple months away."
        )
    if m == 1:
        return (
            "New year, but school is in annual-exam prep crunch mode. "
            "Cold mornings, foggy bus rides."
        )
    if m == 2:
        return (
            "Annual exams happening or wrapping up. Maximum stress. "
            "Saraswati Puja is the only fun thing this month."
        )
    if m == 3:
        return (
            "End of the school year. Exams done, results impending. "
            "Mentally already on summer break. The current grade is about to end."
        )
    return ""


def build_temporal_context(today: date | None = None) -> str:
    """Compose the live 'where in the calendar are we?' block for the AI."""
    today = today or date.today()
    grade = GRADE_START_LEVEL + _academic_year_offset(today)
    grade_ord = _ordinal(grade)
    day_name = today.strftime("%A")
    pretty_date = (
        today.strftime("%B %-d, %Y")
        if os.name != "nt"
        else today.strftime("%B %#d, %Y")
    )

    return (
        "\n\n## Temporal Context (live, auto-updated each run)\n"
        f"- Today is **{day_name}, {pretty_date}**.\n"
        f"- You are currently in **{grade_ord} grade** at TIGPS Nabagram.\n"
        f"- {_school_phase(today)}\n"
        "- The West Bengal academic year runs **April → March**, so April 1 is when "
        "you get promoted to the next grade.\n"
        "- Reference the date / season / school phase only when it's naturally "
        "relevant. Don't open every message with the date — that's weird.\n"
    )


def build_system_prompt(
    base: str, personality: str, brain_type: str, user_name: str = ""
) -> str:
    """Assemble the per-turn system prompt.

    Layout matters for Groq's automatic prompt caching: the cache hits on
    matching PREFIXES, breaking at the first byte of difference. So we
    front-load everything that stays identical across turns and push
    per-turn / per-day variability to the end. Concretely:

        [BASE FILE (static within session)]
        [PERSONA SUFFIX (changes only when user switches mode)]
        [THINKER SUFFIX (changes only when brain switches)]
        [TOOL_GUIDANCE (fully static)]
        ─── cacheable prefix ends here on a typical turn ───
        [USER NAME + LORE (changes when name/lore updates)]
        [POPUP STATE (changes per turn while popup active)]
        [TEMPORAL CONTEXT (changes daily)]

    Putting temporal context at the END instead of the start (where it
    used to be) means yesterday's cached prefix can still be re-used
    today for everything before that section. Same idea for the popup
    state — moving it past the lore block keeps lore-tier hits stable
    even when the popup flips state mid-session.
    """
    prompt = base or ""
    prompt += PERSONALITY_SUFFIX.get(personality, "")
    if brain_type == "Thinker":
        prompt += " Use deep thinking to analyze the request before answering."
    prompt += TOOL_GUIDANCE

    if user_name:
        prompt += (
            f"\n\nThe person you are currently chatting with is **{user_name}**. "
            f"You ALREADY KNOW their name — do NOT call `request_user_name`. "
            f"Use their name directly when it feels natural.\n"
        )
        lore_block = lore_store.render_lore_block(user_name)
        if lore_block:
            prompt += "\n" + lore_block + "\n"
    else:
        # If the popup was already shown and declined this session, tell the
        # LLM not to keep asking. Streamlit's session_state is the source of
        # truth here — read at prompt-build time so it's always current.
        try:
            dismissed = bool(st.session_state.get("_name_popup_dismissed"))
            count = int(st.session_state.get("_name_popup_count", 0))
        except Exception:
            dismissed, count = False, 0

        if dismissed or count >= 2:
            prompt += (
                "\n\nYou don't know the user's name and they have already chosen "
                "not to share it. Do NOT call `request_user_name`. Continue "
                "the conversation without using a name.\n"
            )
        elif count >= 1:
            prompt += (
                "\n\nA popup asking for the user's name is already open or was "
                "shown this turn. Do NOT call `request_user_name` again. If "
                "they answer, the name will appear next turn automatically.\n"
            )
        else:
            prompt += (
                "\n\nYou do not yet know the user's name. ONLY if it feels "
                "genuinely natural (not on the first reply, not as a forced "
                "interrogation), you MAY call `request_user_name` exactly "
                "ONCE this session. Never call it twice.\n"
            )

    # Temporal context goes LAST so the daily-changing date string doesn't
    # invalidate the cacheable prefix above it. Previously this was prepended
    # to the base — that meant every new day broke Groq's prompt cache for
    # the entire prompt. Moving it to the end keeps ~95% of the prompt
    # cacheable across the day boundary.
    prompt += build_temporal_context()
    return prompt


# ---------------------------------------------------------------------------
# Brain (LLM) interaction with tool-calling
# ---------------------------------------------------------------------------

MAX_TOOL_HOPS = 4

# Matches the broken tool-call format some Groq / Gemini models emit inside
# `content` instead of using the proper `tool_calls` / `function_call` field.
# Variants observed in the wild (all handled):
#   <function=remember_lore>{"user_name":"Sam","fact":"likes tea"}</function>
#   <function=name:remember_lore {"user_name":"Sam","fact":"likes tea"}</function>
#   <function=name:remember_lore>{"user_name":"Sam","fact":"likes tea"}</function>
#   <function name="remember_lore">{"user_name":"Sam"}</function>
# The body is anything containing a JSON object; we extract the first {...} we
# can balance-parse, ignoring whatever junk separates the name from the args.
_INLINE_TOOLCALL_RE = re.compile(
    r"""<function\s*                              # opening tag, allow whitespace
        (?:=|\s)\s*                               # = or whitespace separator
        (?:name\s*[:=]\s*)?                       # optional 'name:' / 'name=' prefix
        ["']?([a-zA-Z_][a-zA-Z0-9_]*)["']?        # 1: function name (maybe quoted)
        \s*>?\s*                                  # optional closing '>'
        (\{.*?\})                                 # 2: JSON args (non-greedy)
        \s*</function\s*>""",
    re.DOTALL | re.VERBOSE,
)


def _extract_inline_tool_calls(content: str) -> tuple[list[dict], str]:
    """Pull pseudo-XML tool calls out of a model's text content.

    Returns (parsed_calls, cleaned_text). `parsed_calls` mimics the shape of
    OpenAI/Groq `tool_calls` so the rest of the loop can treat them uniformly.
    """
    if not content or "<function" not in content:
        return [], content
    parsed: list[dict] = []
    for i, m in enumerate(_INLINE_TOOLCALL_RE.finditer(content)):
        name = m.group(1)
        raw_args = m.group(2)
        try:
            json.loads(raw_args)  # validate
            args_str = raw_args
        except Exception:
            args_str = "{}"
        parsed.append(
            {
                "id": f"inline_{i}_{int(time.time() * 1000)}",
                "name": name,
                "arguments": args_str,
            }
        )
    cleaned = _INLINE_TOOLCALL_RE.sub("", content).strip()
    return parsed, cleaned


def _strip_inline_tool_noise(text: str) -> str:
    """Remove any leftover `<function=...>...</function>` pseudo-XML from
    text that's about to be shown to the user. Used as a last-line defence
    when a model emits the blob but we already finished tool-calling hops.
    """
    if not text or "<function" not in text:
        return text
    cleaned = _INLINE_TOOLCALL_RE.sub("", text)
    # Also nuke malformed/stray tags that didn't match the full pattern
    # (e.g. unterminated or missing JSON body).
    cleaned = re.sub(r"</?function[^>]*>", "", cleaned)
    return cleaned.strip()


def _record_cache_usage(response: Any, *, model: str = "", hop: str = "") -> None:
    """Pull Groq's prompt-cache hit metrics off a chat completion response
    and stash them in session_state so the sidebar can show savings.

    Groq returns `usage.prompt_tokens_details.cached_tokens` on every
    completion. A hit means we paid 50% for those tokens instead of full
    price (their automatic caching, no opt-in needed — we just need the
    prefix to stay deterministic, which is what the prompt-building
    refactor above is for).

    Stored shape (session_state['_groq_cache_stats']):
        {
          'total_prompt_tokens': int,   # cumulative across session
          'total_cached_tokens': int,   # cumulative across session
          'last_prompt_tokens': int,
          'last_cached_tokens': int,
          'last_model': str,
          'last_hop': str,
        }
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = 0
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        # On the openai SDK shape this may be a dict instead of an object.
        if cached == 0:
            d = (
                getattr(usage, "prompt_tokens_details", None)
                if not isinstance(usage, dict)
                else usage.get("prompt_tokens_details")
            )
            if isinstance(d, dict):
                cached = int(d.get("cached_tokens", 0) or 0)

        stats = st.session_state.setdefault("_groq_cache_stats", {
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "last_prompt_tokens": 0,
            "last_cached_tokens": 0,
            "last_model": "",
            "last_hop": "",
        })
        stats["total_prompt_tokens"] += int(prompt_tokens)
        stats["total_cached_tokens"] += int(cached)
        stats["last_prompt_tokens"] = int(prompt_tokens)
        stats["last_cached_tokens"] = int(cached)
        stats["last_model"] = model
        stats["last_hop"] = hop
    except Exception:
        # Cache telemetry must never break a turn.
        pass


def _stream_groq_final(
    model: str, messages: list, timeout_seconds: float
) -> Iterator[str]:
    """Yield text chunks for the FINAL (no-tools) Groq completion.

    Used after any tool-calling hops have resolved, so we can stream the
    user-visible answer for faster perceived latency.

    `stream_options={"include_usage": True}` tells Groq to emit a final
    usage chunk after the content stream completes — that's where the
    prompt-cache hit stats live. Without it we can't measure cache wins
    on streamed turns.
    """
    stream = groq_client.chat.completions.create(
        messages=messages,
        model=model,
        # Don't pass `tools` here; with tool_choice="none" some models still
        # echo the schema or emit empty content. Plain prose mode is safer.
        timeout=timeout_seconds,
        max_completion_tokens=1500,
        stream=True,
        stream_options={"include_usage": True},
    )
    last_chunk = None
    for chunk in stream:
        last_chunk = chunk
        try:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                # Usage-only final chunk has empty choices on most providers.
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
        except Exception:
            piece = None
        if piece:
            yield piece
    # Record cache stats from the final chunk's usage block (if present).
    if last_chunk is not None:
        _record_cache_usage(last_chunk, model=model, hop="stream-final")


def _groq_chat_with_tools(model: str, messages: list, timeout_seconds: float) -> str:
    """Run a Groq chat completion with a bounded tool-calling loop.

    Strategy:
      1. First hop is non-streaming with tool_choice="auto" so we can inspect
         tool_calls (streaming + tools is messy across SDKs).
      2. If structured tool_calls come back -> run them, then re-loop.
      3. If the model emits the broken <function=...> pseudo-XML inside
         content -> parse it ourselves, treat as real tool calls, re-loop.
      4. Once no tools are pending, stream the final answer for low latency.
    """
    for hop in range(MAX_TOOL_HOPS):
        kwargs = dict(
            messages=messages,
            model=model,
            timeout=timeout_seconds,
            max_completion_tokens=1500,
        )
        if hop == 0:
            # Only offer tools on the first hop; subsequent hops are pure prose.
            kwargs["tools"] = get_openai_tools_cached()
            kwargs["tool_choice"] = "auto"
        response = groq_client.chat.completions.create(**kwargs)
        _record_cache_usage(response, model=model, hop=f"chat-with-tools/hop-{hop}")
        try:
            msg = response.choices[0].message
        except Exception:
            return str(response) or "No response generated."

        content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Normalise structured tool calls into a common shape.
        normalised: list[dict] = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            }
            for tc in tool_calls
        ]

        # Fallback: parse inline <function=...>{...}</function> blobs that some
        # models (gpt-oss-*) wrongly put inside `content` instead of tool_calls.
        if not normalised:
            inline_calls, cleaned_content = _extract_inline_tool_calls(content)
            if inline_calls:
                print(
                    f"[Groq] {model} emitted {len(inline_calls)} inline pseudo "
                    f"tool call(s); recovering.",
                    file=sys.stderr,
                )
                normalised = inline_calls
                content = cleaned_content

        if not normalised:
            # Plain text answer — done. (Already non-streaming here; streaming
            # only buys us time when we KNOW there were no tool calls to make,
            # which we can only know after this first call.)
            return _strip_inline_tool_noise(content) or "No response generated."

        # Record assistant turn with the tool calls it made.
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in normalised
                ],
            }
        )
        # Execute every requested tool IN PARALLEL on the shared pool and
        # feed results back. Sequential dispatch used to be the dominant
        # latency contributor when the model emitted 2–3 tool calls per
        # hop (each ~1–2s of Firestore I/O).
        for tc in ai_tools.dispatch_parallel(normalised):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": tc["content"],
                }
            )

    # If we got here, we hit MAX_TOOL_HOPS without a clean final answer.
    # Make one last streaming pass with tools disabled to force prose output.
    try:
        chunks = list(_stream_groq_final(model, messages, timeout_seconds))
        final = "".join(chunks).strip()
        if final:
            return final
    except Exception as e:
        print(f"[Groq] final stream failed on {model}: {e}", file=sys.stderr)
    return "Hmm, I got tangled up calling tools. Try asking again."


def _gemini_chat_with_tools(
    model: str,
    system_prompt: str,
    full_prompt: str,
    temperature: float,
    timeout_seconds: float,
) -> str:
    """Run a Gemini call with a bounded function-calling loop."""
    contents = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]
    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        # No `thinking_config` here: forcing a thinking budget makes flash
        # models slow without quality gain for casual chat. The Thinker mode
        # picks reasoning-capable models in the sidebar instead.
        temperature=temperature,
        tools=get_gemini_tools_cached() or None,
        # Gemini SDK takes timeout in *milliseconds*. Don't undershoot — the
        # 2.5/3.x flash models routinely take 6–15s with a 17kB system prompt.
        http_options=types.HttpOptions(timeout=int(max(timeout_seconds, 25) * 1000)),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"
            ),
        ],
    )

    for hop in range(MAX_TOOL_HOPS):
        # Disable tools on subsequent hops to force a final text response.
        if hop > 0:
            cfg.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            )

        response = gemini_client.models.generate_content(
            model=model,
            config=cfg,
            contents=contents,
        )

        fcalls = []
        try:
            for cand in response.candidates or []:
                for part in cand.content.parts or []:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        fcalls.append(fc)
        except Exception:
            pass

        if not fcalls:
            text = getattr(response, "text", None)
            if text and text.strip():
                # Some Gemini flash builds (esp. 3.x previews) occasionally emit
                # tool calls as inline `<function=...>{...}</function>` text
                # instead of structured function_call parts. If we spot one,
                # execute it ourselves and continue the loop instead of leaking
                # the raw pseudo-XML to the user.
                inline_calls, cleaned_text = _extract_inline_tool_calls(text)
                if inline_calls:
                    print(
                        f"[Gemini] {model} emitted {len(inline_calls)} inline "
                        f"pseudo tool call(s); recovering."
                    )
                    try:
                        contents.append(response.candidates[0].content)
                    except Exception:
                        pass
                    # Run recovered tool calls in parallel — same speedup
                    # win as the structured-fcall path below.
                    for tc in ai_tools.dispatch_parallel(inline_calls):
                        contents.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=tc["name"],
                                        response=tc["result"],
                                    )
                                ],
                            )
                        )
                    continue
                return _strip_inline_tool_noise(text)
            # Empty text -> raise so outer loop tries the next model.
            raise RuntimeError(f"{model} returned empty text")

        try:
            contents.append(response.candidates[0].content)
        except Exception:
            pass
        # Normalise the Gemini-shaped fcalls into the parallel-dispatcher
        # contract, then fire them concurrently. Most lore lookups hit
        # Firestore — running them sequentially used to double the round-
        # trip cost when the model invoked recall + remember in one hop.
        normalised = [
            {
                "id": f"gemini_{i}",
                "name": fc.name,
                "arguments": json.dumps(dict(fc.args) if getattr(fc, "args", None) else {}),
            }
            for i, fc in enumerate(fcalls)
        ]
        for tc in ai_tools.dispatch_parallel(normalised):
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tc["name"], response=tc["result"],
                        )
                    ],
                )
            )
    return "Hmm, I got tangled up calling tools. Try asking again."


def _format_brain_error(brain: str, attempts: list[tuple[str, Exception]]) -> str:
    """Build a user-visible error that actually says what broke."""
    if not attempts:
        return f"Sorry, I couldn't reach {brain}. No models were even attempted."
    last_model, last_err = attempts[-1]
    err_str = str(last_err)
    summary = f"❌ All {brain} models failed.\n\nLast error on `{last_model}`:\n`{type(last_err).__name__}: {last_err}`"
    if len(attempts) > 1:
        tried = ", ".join(f"`{m}`" for m, _ in attempts)
        summary += f"\n\nTried in order: {tried}"

    hints = []
    # 404 / not-found must be checked FIRST. The old order let a "model not
    # found" error whose body happened to mention rate_limit get classified
    # as a TPM issue — that's exactly what made the model changer look like
    # it was producing rate-limit errors.
    not_found = (
        "404" in err_str
        or "not exist" in err_str.lower()
        or "model_not_found" in err_str.lower()
        or "model_decommissioned" in err_str.lower()
        or "does not exist" in err_str.lower()
    )
    if not_found:
        hints.append(
            "**Model ID not found / decommissioned.** Open the sidebar → "
            "Model Fallback Chains and pick a different model. The picker "
            "lists every currently-available model for your API key."
        )
    elif "rate_limit" in err_str.lower() or "Limit 6000" in err_str or "TPM" in err_str:
        hints.append(
            "**Groq tokens-per-minute cap.** The system prompt is ~17kB. Either:\n"
            "  - Wait 60s and try again, or\n"
            "  - Drop `llama-3.1-8b-instant` from the chain (it's a 6K-TPM model), or\n"
            "  - Use `llama-3.3-70b-versatile` (300K TPM), or\n"
            "  - Upgrade at https://console.groq.com/settings/billing"
        )
    if "timeout" in err_str.lower() or "TimeoutError" in type(last_err).__name__:
        hints.append("**Timeout.** Bump the per-model timeout slider in the sidebar.")
    if "401" in err_str or "unauthorized" in err_str.lower():
        hints.append(f"**Auth failed.** Check your `{brain.upper()}_API_KEY` value.")

    if not hints:
        hints.append(
            "Check the env vars, the model IDs in the sidebar, and the per-model timeout."
        )
    summary += "\n\n**Likely cause / fix:**\n" + "\n\n".join(hints)
    return summary


def _groq_first_hop_discover(
    model: str, messages: list, timeout_seconds: float
) -> tuple[list, str, list[dict]]:
    """Run the non-streaming first hop. Returns `(messages, prose, calls)`.

    `prose` is whatever text the assistant emitted alongside its tool calls
    (often a short "Sure, let me look that up…" — we want to yield it to
    the user *immediately* while tools run in the background).
    `calls` is the normalised list of tool calls to execute, or [] if the
    model is done.

    Raises on API/timeout errors so the outer fallback loop can pick a
    different model.
    """
    response = groq_client.chat.completions.create(
        messages=messages,
        model=model,
        tools=get_openai_tools_cached(),
        tool_choice="auto",
        timeout=timeout_seconds,
        max_completion_tokens=1500,
    )
    _record_cache_usage(response, model=model, hop="first-hop-discover")
    msg = response.choices[0].message
    content = msg.content or ""
    tool_calls = getattr(msg, "tool_calls", None) or []
    normalised = [
        {
            "id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments or "{}",
        }
        for tc in tool_calls
    ]
    if not normalised:
        inline_calls, cleaned = _extract_inline_tool_calls(content)
        if inline_calls:
            normalised = inline_calls
            content = cleaned

    if not normalised and not content.strip():
        # Empty content AND no tools — raise so the outer loop falls back
        # to the next model in the chain.
        raise RuntimeError(f"{model} returned empty content")

    return messages, _strip_inline_tool_noise(content), normalised


def _groq_stream_with_parallel_tools(
    model: str, messages: list, prose: str, calls: list[dict], timeout_seconds: float,
) -> Iterator[str]:
    """Post-first-hop generator: yield prose immediately, run tools in
    parallel, then stream the synthesis.

    Old behaviour: first hop (blocking) → tools (blocking, sequential) →
    stream final answer. User saw a spinner for 10+ seconds when tools
    were involved.

    New behaviour:
      1. Caller has already done the first hop (so timeouts surfaced
         before any token was yielded — the outer fallback loop works).
      2. We yield the model's leading prose RIGHT NOW (instant feedback).
      3. We fire every tool call on the shared thread pool — they execute
         in PARALLEL. Most tools are I/O bound (Firestore, SQLite), so
         this collapses N tools' latency to max(latency_i) instead of sum.
      4. When tools resolve, we stream the second hop (final synthesis).

    The result: user sees text within ~500ms of pressing send, even if the
    full reply requires several seconds of tool work.
    """
    # ---- Optimistic update: paint the prose BEFORE running tools ----
    # If the model produced any leading text ("Hold on, let me check…"),
    # show it to the user right away. This is the entire UX win — the
    # chat starts moving while tools work in the background.
    if prose:
        yield prose
        # Visually separate the prose from the streaming synthesis that
        # follows so they read as two paragraphs rather than a run-on.
        yield "\n\n"

    # ---- Append the assistant turn (with its tool_calls) ----
    messages.append(
        {
            "role": "assistant",
            "content": prose,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in calls
            ],
        }
    )

    # ---- Run all tools in parallel on the shared pool ----
    # `dispatch_parallel` returns results in input order with `content`
    # (the JSON-serialised result) already filled in. The tool_status
    # banner fragment will paint "🔧 running X…" lines while this blocks.
    completed = ai_tools.dispatch_parallel(calls)
    for tc in completed:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": tc["content"],
            }
        )

    # ---- Stream the final synthesis ----
    yield from _stream_groq_final(model, messages, timeout_seconds)


def _groq_run_tools_then_stream(
    model: str, messages: list, timeout_seconds: float
) -> Iterator[str]:
    """Legacy entry point preserved for compatibility — discovers tools then
    delegates to `_groq_stream_with_parallel_tools`. New code paths in
    `get_ai_response_with_brain` call the two halves separately so the
    discover hop's exceptions surface before any token is yielded.
    """
    messages, prose, calls = _groq_first_hop_discover(
        model, messages, timeout_seconds
    )
    if not calls:
        if prose:
            yield prose
        return
    yield from _groq_stream_with_parallel_tools(
        model, messages, prose, calls, timeout_seconds
    )


def get_ai_response_with_brain(
    prompt: str,
    system_prompt: str,
    brain_type: str,
    chat_history: list,
    temperature: float,
    *,
    stream: bool = False,
):
    """Call the selected brain (Fast=Groq, Thinker=Gemini) with model fallback.

    With stream=True (Fast brain only), returns an Iterator[str] of text chunks
    instead of a single string, so the UI can render tokens as they arrive.
    """
    # Realistic default: 70B Groq models with tool calling + ~17kB system
    # prompt routinely take 5–8s on first hit. Gemini "thinking" models can
    # take 10–20s. Anything under 20s is asking for spurious timeouts.
    timeout = getattr(st.session_state, "fallback_timeout", 25)

    if brain_type == "Fast":
        if groq_client is None:
            return "❌ Groq client not initialized. Set `GROQ_API_KEY` in your environment."
        base = [{"role": "system", "content": system_prompt}]
        for m in chat_history[-6:]:
            if m.get("role") in ("user", "assistant"):
                base.append({"role": m["role"], "content": m.get("content", "")})
        base.append({"role": "user", "content": prompt})

        attempts: list[tuple[str, Exception]] = []
        models = getattr(st.session_state, "groq_models", DEFAULT_GROQ_MODELS)
        for model in models:
            try:
                if stream:
                    # Run the first (non-streaming) hop synchronously HERE so
                    # API/timeout errors raise BEFORE the UI starts consuming
                    # the stream — otherwise the outer fallback loop never
                    # sees the exception. The new `_groq_first_hop_discover`
                    # only blocks on the LLM call itself, not on tools, so
                    # the wait is the same as a no-tools chat.
                    msgs = list(base)
                    msgs, prose, calls = _groq_first_hop_discover(
                        model, msgs, timeout
                    )
                    if not calls:
                        # No tools needed — yield the prose as a one-shot
                        # iterator so the caller still gets the streaming API.
                        return iter([prose])
                    # Tools needed — return the parallel-tools-+-stream
                    # generator. It yields the model's leading prose
                    # IMMEDIATELY, then fires tools in parallel, then
                    # streams the synthesis. Errors mid-stream surface via
                    # display_and_store_response's fallback markdown.
                    return _groq_stream_with_parallel_tools(
                        model, msgs, prose, calls, timeout
                    )
                return _groq_chat_with_tools(model, list(base), timeout)
            except Exception as e:
                attempts.append((model, e))
                # Print to server log so you can grep the traceback there too.
                print(
                    f"[Groq] {model} failed: {type(e).__name__}: {e}", file=sys.stderr
                )
                continue
        return _format_brain_error("Groq", attempts)

    if brain_type == "Thinker":
        if gemini_client is None:
            return "❌ Gemini client not initialized. Set `GOOGLE_API_KEY` in your environment."
        # Filter out `system_note` rows (inline lore-save confirmations) —
        # they're UI-only and would just confuse the model if we fed them
        # back as conversation context.
        ctx_parts = [
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content')}"
            for m in chat_history[-6:]
            if m.get("role") in ("user", "assistant")
        ]
        full_prompt = "\n\n".join(ctx_parts + [f"User: {prompt}"])

        attempts: list[tuple[str, Exception]] = []
        models = getattr(st.session_state, "gemini_models", DEFAULT_GEMINI_MODELS)
        for model in models:
            try:
                return _gemini_chat_with_tools(
                    model, system_prompt, full_prompt, temperature, timeout
                )
            except Exception as e:
                attempts.append((model, e))
                print(
                    f"[Gemini] {model} failed: {type(e).__name__}: {e}", file=sys.stderr
                )
                continue
        return _format_brain_error("Gemini", attempts)

    return "Invalid brain type selected."


# ---------------------------------------------------------------------------
# Name popup
# ---------------------------------------------------------------------------


@_fragment
def _maybe_show_name_popup() -> None:
    """Show the name-request popup when the AI has triggered it.

    Wrapped in @st.fragment so a 'Save' click only reruns this dialog —
    the main chat container, sidebar, and model picker are untouched.
    Previously this called st.rerun() which forced a full-page redraw
    and made the popup feel laggy on slow connections.
    """
    if not st.session_state.get("_name_popup_pending"):
        return
    # Belt-and-braces: if the name is already on file, the popup must NEVER
    # show. This catches the edge case where the LLM raised the popup in
    # one turn and the user typed their name into the sidebar instead.
    if (st.session_state.get("user_name") or "").strip():
        st.session_state["_name_popup_pending"] = False
        return

    reason = (
        st.session_state.get("_name_popup_reason")
        or "So I can remember stuff about you."
    )
    dialog_decorator = getattr(st, "dialog", None) or getattr(
        st, "experimental_dialog", None
    )

    def _render_form() -> None:
        st.write(reason)
        with st.form("name_popup_form", clear_on_submit=False):
            # Stable widget `key=` so typing isn't clobbered by a fragment
            # rerun. Without a key, Streamlit recreates the widget on every
            # rerun and re-seeds the value from `value=`, losing in-flight
            # keystrokes if the user types fast.
            name = st.text_input(
                "Your name",
                value=st.session_state.get("user_name", ""),
                key="_name_popup_text_input",
                placeholder="Type your name here…",
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                save_clicked = st.form_submit_button("💾 Save", use_container_width=True)
            with col2:
                skip_clicked = st.form_submit_button(
                    "Skip for now", use_container_width=True, type="secondary",
                )

            if save_clicked and name.strip():
                # Persist the name + clear popup state in one shot so the
                # next script rerun sees a clean slate.
                # NOTE: do NOT assign to st.session_state._name_popup_text_input
                # here — Streamlit raises StreamlitAPIException when you mutate
                # a widget's bound session_state key after the widget has been
                # instantiated this run. The form is about to be torn down
                # anyway (popup_pending=False), so there's nothing to clear.
                st.session_state.user_name = name.strip()
                lore_store.ensure_user(name.strip())
                st.session_state._name_popup_pending = False
                st.session_state._name_popup_reason = ""
                # Force a FULL rerun (not fragment-scoped) — we need the
                # chat container's next system_prompt build to pick up the
                # new name, which is in the outer script's closure.
                st.rerun()
            elif skip_clicked:
                # User explicitly declined. Lock the popup down so the LLM
                # can't keep firing it. They can still type into the sidebar
                # later if they change their mind.
                st.session_state._name_popup_pending = False
                st.session_state._name_popup_reason = ""
                st.session_state._name_popup_dismissed = True
                st.rerun()

    if dialog_decorator is not None:

        @dialog_decorator("👋 What should I call you?")
        def _popup():
            _render_form()

        _popup()
    else:
        with st.container(border=True):
            st.subheader("👋 What should I call you?")
            _render_form()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

TTS_ENGINE_MAP = {
    "Edge TTS (Free)": "edge",
    "Google TTS (Free)": "gtts",
    "Sarvam.ai": "sarvam",
    "Fish Audio": "fish_audio",
    "SiliconFlow": "silicon_flow",
}


def _sidebar_identity() -> None:
    st.sidebar.markdown("**Who am I talking to?**")
    typed = st.sidebar.text_input(
        "Your name",
        value=st.session_state.get("user_name", ""),
        key="user_name_input",
        placeholder="Tell me your name…",
    )
    if typed and typed != st.session_state.get("user_name", ""):
        st.session_state.user_name = typed.strip()
        lore_store.ensure_user(typed.strip())

    if st.session_state.get("user_name"):
        name = st.session_state.user_name
        public_facts = lore_store.list_public_facts(name)
        private_facts = lore_store.list_private_facts(name)

        with st.sidebar.expander(
            f"🌐 Public lore for {name} ({len(public_facts)})",
            expanded=False,
        ):
            st.caption(
                "Harmless preferences — saved to `lore.json`, visible to anyone using this app."
            )
            if public_facts:
                for f in public_facts:
                    st.markdown(f"- {f}")
            else:
                st.caption("Nothing public yet.")

        with st.sidebar.expander(
            f"🔒 Private lore for {name} ({len(private_facts)})",
            expanded=False,
        ):
            backend = (
                "Firebase"
                if getattr(lore_store, "_use_firebase", False)
                else "local SQLite"
            )
            st.caption(f"Sensitive info — saved to {backend}, never shown publicly.")
            if private_facts:
                for f in private_facts:
                    st.markdown(f"- {f}")
            else:
                st.caption("Nothing private yet.")


def _sidebar_personality_and_brain() -> tuple[str, str]:
    st.sidebar.markdown("**Personality**")
    personality = st.sidebar.selectbox(
        "Select Personality",
        (
            "Roaster",
            "Smart",
            "Debater",
            "Strategic",
            "Tech Nerd",
            "Chill Squad",
            "Exhausted Student",
        ),
        key="personality_selector",
        label_visibility="collapsed",
    )
    from styles import apply_theme

    apply_theme(personality)
    st.sidebar.caption(PERSONALITY_CAPTION.get(personality, ""))

    st.sidebar.markdown("**Brain Power**")
    brain_type = st.sidebar.selectbox(
        "Select Brain",
        ("Fast", "Thinker"),
        key="brain_selector",
        label_visibility="collapsed",
    )
    st.sidebar.caption(
        "⚡ **Fast:** Instant answers (Groq)"
        if brain_type == "Fast"
        else "🕵️ **Thinker:** Deep reasoning (Gemini 2.5 Thinking)"
    )
    return personality, brain_type


def _sidebar_model_settings() -> float:
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_spoken_idx = -1
        st.session_state.greeting_shown = False
        st.rerun()

    temperature = st.sidebar.slider("Creativity Level (Chaos)", 0.0, 1.0, 0.7, 0.1)

    st.sidebar.markdown("**Fallback Settings**")
    st.session_state.fallback_timeout = st.sidebar.slider(
        "Per-model timeout (seconds)",
        5,
        60,
        25,
        1,
        help="How long to wait for each model before falling back to the next. "
        "Lower values keep the UI responsive when other tools are running tests.",
    )

    _sidebar_model_chain_picker()
    _sidebar_cache_savings()
    return temperature


def _sidebar_cache_savings() -> None:
    """Show Groq prompt-cache hit stats so you can verify the cost
    optimisation is actually working. Stats are populated by
    `_record_cache_usage` after every Groq completion in this session.

    Why this exists: Groq's prompt caching is automatic but invisible.
    The only way to know it's hitting is to read `cached_tokens` off the
    usage block. Surfacing it in the sidebar turns "I think we saved
    money" into a measurable number you can quote.
    """
    stats = st.session_state.get("_groq_cache_stats")
    if not stats or stats.get("total_prompt_tokens", 0) == 0:
        return

    total = stats["total_prompt_tokens"]
    cached = stats["total_cached_tokens"]
    last_total = stats["last_prompt_tokens"]
    last_cached = stats["last_cached_tokens"]

    hit_rate = (cached / total * 100) if total else 0
    last_hit_rate = (last_cached / last_total * 100) if last_total else 0
    # Groq's documented discount is 50% on cached input tokens — so the
    # effective tokens billed = total - 0.5*cached.
    effective_billed = total - 0.5 * cached
    savings_pct = ((total - effective_billed) / total * 100) if total else 0

    with st.sidebar.expander(f"💰 Cache savings: {savings_pct:.0f}%", expanded=False):
        st.caption(
            f"**Session total:** {total:,} prompt tokens, "
            f"{cached:,} from cache ({hit_rate:.0f}% hit rate)."
        )
        st.caption(
            f"**Last turn:** {last_total:,} tokens, "
            f"{last_cached:,} cached ({last_hit_rate:.0f}% hit rate)."
        )
        st.caption(
            f"Effective billed: ~{effective_billed:,.0f} tokens "
            f"(saved ~{total - effective_billed:,.0f} = {savings_pct:.0f}%)."
        )
        st.caption(
            "Groq applies a 50% discount on cached input tokens automatically. "
            "Cache expires after ~2 hours of no use."
        )


def _sidebar_model_chain_picker() -> None:
    """Picker UI for the per-provider fallback chains.

    Uses a `multiselect` (validated against the live catalogue) instead of a
    free-form text input — typos in the old text box used to 404 every
    model, which then made the outer fallback loop hit the low-TPM Groq
    model and report a misleading 'rate_limit' error.

    A small text field below the picker lets power users register a custom
    model ID that isn't in the live listing (preview releases, private
    deployments, etc).
    """
    st.sidebar.markdown("**Model Fallback Chains**")
    st.sidebar.caption("Tried in order. Each model is attempted before the next.")

    # ----- Groq -----
    groq_catalogue = fetch_groq_catalogue()
    # Merge in any custom IDs the user added in a previous rerun so they
    # remain selectable.
    custom_groq = st.session_state.get("_custom_groq_models", [])
    groq_options = list(dict.fromkeys(list(groq_catalogue) + list(custom_groq)))

    current = st.session_state.get("groq_models") or list(DEFAULT_GROQ_MODELS)
    current = [m for m in current if m in groq_options]
    if not current:
        current = [m for m in DEFAULT_GROQ_MODELS if m in groq_options]

    chosen_groq = st.sidebar.multiselect(
        "Groq models (Fast brain)",
        options=groq_options,
        default=current,
        key="groq_models_picker",
        help="Drag the chips left-to-right to reorder. The first one is tried first.",
    )
    if chosen_groq:
        st.session_state.groq_models = chosen_groq
    else:
        st.sidebar.caption("⚠️ No Groq model selected — falling back to defaults.")
        st.session_state.groq_models = list(DEFAULT_GROQ_MODELS)

    # Surface a warning if the user keeps a known low-TPM model in their chain.
    low_tpm_selected = [
        m for m in st.session_state.groq_models if m in LOW_TPM_GROQ_MODELS
    ]
    if low_tpm_selected:
        for m in low_tpm_selected:
            st.sidebar.warning(f"`{m}`: {LOW_TPM_GROQ_MODELS[m]}")

    # ----- Gemini -----
    gemini_catalogue = fetch_gemini_catalogue()
    custom_gemini = st.session_state.get("_custom_gemini_models", [])
    gemini_options = list(dict.fromkeys(list(gemini_catalogue) + list(custom_gemini)))

    current_gem = st.session_state.get("gemini_models") or list(DEFAULT_GEMINI_MODELS)
    current_gem = [m for m in current_gem if m in gemini_options]
    if not current_gem:
        current_gem = [m for m in DEFAULT_GEMINI_MODELS if m in gemini_options]

    chosen_gem = st.sidebar.multiselect(
        "Gemini models (Thinker brain)",
        options=gemini_options,
        default=current_gem,
        key="gemini_models_picker",
    )
    if chosen_gem:
        st.session_state.gemini_models = chosen_gem
    else:
        st.sidebar.caption("⚠️ No Gemini model selected — falling back to defaults.")
        st.session_state.gemini_models = list(DEFAULT_GEMINI_MODELS)

    # ----- Custom model escape hatch -----
    with st.sidebar.expander("➕ Add a custom model ID", expanded=False):
        st.caption(
            "For preview releases or private deployments not in the live catalogue."
        )
        provider = st.radio(
            "Provider", ("Groq", "Gemini"), horizontal=True, key="_custom_provider"
        )
        custom_id = st.text_input(
            "Model ID",
            key="_custom_model_input",
            placeholder="e.g. gemini-4-flash-preview",
        )
        if st.button("Add", key="_custom_model_add"):
            mid = (custom_id or "").strip()
            if not mid:
                st.warning("Empty model ID ignored.")
            else:
                bucket_key = (
                    "_custom_groq_models"
                    if provider == "Groq"
                    else "_custom_gemini_models"
                )
                bucket = st.session_state.setdefault(bucket_key, [])
                if mid in bucket:
                    st.info(f"`{mid}` already in custom list.")
                else:
                    bucket.append(mid)
                    st.success(
                        f"Added `{mid}`. Select it above to include in the chain."
                    )
                    st.rerun()

    # ----- Refresh button -----
    if st.sidebar.button("🔄 Refresh model catalogue"):
        fetch_groq_catalogue.clear()
        fetch_gemini_catalogue.clear()
        st.rerun()


def _sidebar_tts() -> tuple[str, str, str]:
    """Render TTS controls. Returns (engine_label, voice, lang)."""
    st.sidebar.markdown("**TTS Engine**")
    engine_options = ["Edge TTS (Free)", "Google TTS (Free)"]
    if SARVAM_API_KEY:
        engine_options.append("Sarvam.ai")
    if FISH_AUDIO_API_KEY:
        engine_options.append("Fish Audio")
    if SILICON_FLOW_API_KEY:
        engine_options.append("SiliconFlow")

    engine = st.sidebar.selectbox("Select TTS Engine", options=engine_options)
    voice, lang = "default", "en"

    if engine == "Edge TTS (Free)":
        st.sidebar.markdown("**Edge Voice**")
        label = st.sidebar.selectbox(
            "Select Voice", options=list(EDGE_VOICES.keys()), index=0
        )
        voice = EDGE_VOICES[label]

    elif engine == "Google TTS (Free)":
        st.sidebar.markdown("**gTTS Language**")
        langs = {
            "English": "en",
            "Hindi": "hi",
            "Bengali": "bn",
            "Spanish": "es",
            "French": "fr",
        }
        lang = st.sidebar.selectbox(
            "Language",
            options=list(langs.values()),
            format_func=lambda x: next(k for k, v in langs.items() if v == x),
        )

    elif engine == "Sarvam.ai":
        st.sidebar.markdown("**Sarvam Speaker**")
        voice = st.sidebar.selectbox(
            "Select Speaker",
            options=[
                "Shubh",
                "Aditya",
                "Ritu",
                "Priya",
                "Neha",
                "Rahul",
                "Pooja",
                "Rohan",
                "Simran",
                "Kavya",
            ],
        )
        langs = {
            "English (India)": "en-IN",
            "Hindi": "hi-IN",
            "Tamil": "ta-IN",
            "Telugu": "te-IN",
        }
        lang = st.sidebar.selectbox(
            "Language",
            options=list(langs.values()),
            format_func=lambda x: next(k for k, v in langs.items() if v == x),
        )

    elif engine == "Fish Audio":
        st.sidebar.markdown("**Fish Audio Voice**")
        voice = st.sidebar.selectbox(
            "Select Voice",
            options=["default", "e_girl", "young_boy", "mature_female", "male"],
        )
        langs = {"English": "en", "Chinese": "zh", "Spanish": "es", "French": "fr"}
        lang = st.sidebar.selectbox(
            "Language",
            options=list(langs.values()),
            format_func=lambda x: next(k for k, v in langs.items() if v == x),
        )

    elif engine == "SiliconFlow":
        st.sidebar.markdown("**SiliconFlow Voice**")
        voice = st.sidebar.selectbox(
            "Select Voice",
            options=["default", "narrator_en", "narrator_zh", "casual_en", "casual_zh"],
        )

    st.session_state.autoplay_response = st.sidebar.checkbox(
        "🔁 Auto-play response",
        value=st.session_state.get("autoplay_response", False),
        help="Automatically speak every new assistant reply. "
        "Some browsers may require you to click once before audio plays.",
    )
    st.session_state.use_html_autoplay = st.sidebar.checkbox(
        "Use HTML autoplay for audio (single player)", value=False
    )
    st.session_state.compact_audio_icon = st.sidebar.checkbox(
        "Compact audio icon (play/pause)", value=False
    )
    return engine, voice, lang


# ---------------------------------------------------------------------------
# Audio player rendering
# ---------------------------------------------------------------------------


def _render_compact_audio(audio_bytes: bytes) -> None:
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_id = f"audio_{int(time.time() * 1000)}"
    html = f"""
<div>
  <button aria-label="Play"
          style="font-size:20px; border:none; background:transparent; cursor:pointer;"
          onclick="(function(a,b){{var x=document.getElementById(a); if(!x)return; if(x.paused){{x.play(); b.innerText='⏸️';}} else {{x.pause(); b.innerText='▶️';}}}})('{audio_id}', this)">▶️</button>
  <audio id="{audio_id}" src="data:audio/mp3;base64,{b64}" style="display:none;"></audio>
</div>
"""
    components_html(html, height=44)


def _render_html_autoplay(audio_bytes: bytes) -> None:
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_id = f"autoplay_audio_{int(time.time() * 1000)}"
    html = f"""
<audio id="{audio_id}" src="data:audio/mp3;base64,{b64}" style="display:none;"></audio>
<button id="fallback_{audio_id}"
        style="font-size:18px; border:none; background:transparent; cursor:pointer; display:none;"
        onclick="(function(a,b){{var x=document.getElementById(a); if(!x)return; if(x.paused){{x.play(); b.innerText='⏸️';}} else {{x.pause(); b.innerText='▶️';}}}})('{audio_id}', this)">▶️</button>
<script>
  var a = document.getElementById('{audio_id}');
  var b = document.getElementById('fallback_{audio_id}');
  if (a) {{
    a.play().catch(function(){{ if (b) b.style.display = 'inline'; }});
    a.onended = function() {{ if (b) b.innerText = '▶️'; }};
  }}
</script>
"""
    components_html(html, height=44)


def _render_streamlit_audio(audio_bytes: bytes) -> None:
    try:
        buf = io.BytesIO(audio_bytes)
        buf.seek(0)
        st.audio(
            buf,
            format="audio/mp3",
            start_time=0,
            key=f"sanniva_audio_{int(time.time() * 1000)}",
        )
    except Exception:
        try:
            st.audio(audio_bytes, format="audio/mp3")
        except Exception:
            st.warning("Failed to play audio via Streamlit player.")


def render_audio_player(
    audio_bytes: bytes,
    *,
    force_autoplay: bool = False,
    show_download: bool = True,
    show_local_open: bool = True,
) -> None:
    """Render audio. With `force_autoplay`, ignore the compact-icon preference
    and always emit the HTML autoplay player so the browser actually plays it.
    """
    try:
        if force_autoplay:
            _render_html_autoplay(audio_bytes)
        elif st.session_state.get("use_html_autoplay", False):
            _render_html_autoplay(audio_bytes)
        elif st.session_state.get("compact_audio_icon", False):
            _render_compact_audio(audio_bytes)
        else:
            _render_streamlit_audio(audio_bytes)
    except Exception:
        st.warning("Failed to render audio player.")

    if show_download:
        try:
            st.download_button(
                "⬇️ Download audio",
                data=audio_bytes,
                file_name="sanniva_response.mp3",
                mime="audio/mpeg",
                key=f"dl_{int(time.time() * 1000)}",
            )
        except Exception:
            pass

    if show_local_open and st.checkbox(
        "Open local player (server)", key=f"local_{int(time.time() * 1000)}"
    ):
        try:
            play_audio_bytes(audio_bytes)
        except Exception:
            st.warning("Failed to open local player.")


def sanitize_for_tts(text: str) -> str:
    """Strip markdown / formatting so TTS engines read clean prose.

    Without this, every TTS voice literally pronounces the punctuation:
      - `*sighs*`              → "asterisk sighs asterisk"
      - `**bold**`             → "asterisk asterisk bold asterisk asterisk"
      - `` `recall_lore` ``    → "backtick recall underscore lore backtick"
      - `[link](http://x)`     → "bracket link bracket paren http..."

    Strategy: convert to plain text, dropping anything that's purely
    visual formatting. Stage directions like `"*sighs*"` collapse to an
    ellipsis ("…") so there's still a tiny natural pause where the
    persona intended emphasis, instead of the action name being read
    aloud as words.
    """
    if not text:
        return ""

    out = text

    # 1) Code blocks (```...```): drop entirely — they read terribly.
    out = re.sub(r"```[\s\S]*?```", " ", out)

    # 2) Inline code (`x`): keep the inner content but drop the backticks.
    out = re.sub(r"`([^`]+)`", r"\1", out)

    # 3) Bold markers FIRST (before any single-asterisk pass), because
    #    `**bold**` contains `*bold*` as a substring and a naive
    #    single-asterisk regex would eat its inside. Same idea for `__`.
    out = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", out)
    out = re.sub(r"__([^_\n]+)__", r"\1", out)

    # 4) Quoted stage directions ("*sighs*", "*flops on desk*"): collapse
    #    to a short pause so the TTS doesn't read the action name as words.
    out = re.sub(r'"\*[^*"\n]+\*"', "… ", out)
    #    Same pattern without surrounding quotes — but only when the
    #    inside looks like a stage direction (lowercase verb-ish words),
    #    not regular italics. Keep the limit short (≤40 chars).
    out = re.sub(r"\*[^*\n]{1,40}\*", "… ", out)

    # 5) Remaining italic markers (*x*, _x_): keep inner text.
    out = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", out)
    out = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", out)

    # 5) Markdown links [text](url) → just the text.
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)

    # 6) Bare URLs — keep them out of TTS; engines mangle them anyway.
    out = re.sub(r"https?://\S+", "", out)

    # 7) Headings (#, ##, ###) at line start → strip the hashes.
    out = re.sub(r"^\s{0,3}#{1,6}\s*", "", out, flags=re.MULTILINE)

    # 8) Bullet markers (-, *, +) at line start → strip.
    out = re.sub(r"^\s{0,3}[-*+]\s+", "", out, flags=re.MULTILINE)

    # 9) Blockquote markers (>) at line start → strip.
    out = re.sub(r"^\s{0,3}>\s?", "", out, flags=re.MULTILINE)

    # 10) Any stray asterisks/underscores/backticks left over.
    out = out.replace("*", "").replace("`", "")

    # 11) Collapse runs of whitespace.
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)

    return out.strip()


def _tts_for_engine(
    text: str, engine_label: str, voice: str, lang: str
) -> tuple[bytes | None, str]:
    """Resolve engine label + voice to actual audio bytes.

    Strips markdown / stage-direction formatting before handing the text
    to the engine so the voice doesn't pronounce asterisks, backticks,
    etc. literally.
    """
    clean = sanitize_for_tts(text)
    if not clean:
        return None, "❌ Nothing to read aloud (text was empty after sanitisation)"
    engine_key = TTS_ENGINE_MAP.get(engine_label, "edge")
    voice_param = voice.strip().lower() if engine_label == "Sarvam.ai" else voice
    return generate_speech_any(text=clean, engine=engine_key, voice=voice_param, lang=lang)


def _handle_speak_button(
    last_response: str, engine_label: str, voice: str, lang: str
) -> None:
    """Manual 'Speak Response' button for the latest assistant message."""
    if not st.button("🔊 Speak Response"):
        return
    with st.spinner("Generating speech..."):
        audio_bytes, status = _tts_for_engine(last_response, engine_label, voice, lang)
    # Show failures as errors (red) instead of info (blue) so blank-audio
    # bugs are immediately obvious.
    if audio_bytes:
        st.info(status)
        render_audio_player(audio_bytes)
    else:
        st.error(status or "TTS failed with no status message.")


def _maybe_autoplay_response(engine_label: str, voice: str, lang: str) -> None:
    """If auto-play is on and a new assistant message exists, speak it once."""
    if not st.session_state.get("autoplay_response"):
        return

    msgs = st.session_state.get("messages") or []
    if not msgs or msgs[-1].get("role") != "assistant":
        return

    last_idx = len(msgs) - 1
    if last_idx <= st.session_state.get("last_spoken_idx", -1):
        # Already spoken this turn — don't replay on widget reruns.
        return

    text = msgs[last_idx].get("content", "")
    if not text.strip():
        return

    with st.spinner("🔊 Speaking…"):
        audio_bytes, status = _tts_for_engine(text, engine_label, voice, lang)

    # Mark spoken BEFORE rendering so a render-time exception doesn't loop us.
    st.session_state.last_spoken_idx = last_idx

    if audio_bytes:
        # Compact, no download/local-player chrome — just play.
        render_audio_player(
            audio_bytes, force_autoplay=True, show_download=False, show_local_open=False
        )
    else:
        # Surface real failures (network, bad voice ID, missing dep) as an
        # error banner — they used to disappear into a tiny grey caption and
        # the user just saw a missing/blank audio player.
        if status.startswith("❌"):
            st.error(f"TTS failed: {status}")
        else:
            st.caption(f"Auto-play skipped: {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _show_os_greeting() -> None:
    ua = get_user_agent_string()
    os_family = get_os_from_user_agent(ua).lower()
    greeting = OS_GREETING.get(os_family)
    if greeting:
        try:
            st.sidebar.success(greeting)
        except Exception:
            pass


def _show_initial_greeting(personality: str) -> None:
    if st.session_state.greeting_shown:
        return
    greeting = PERSONALITY_GREETING.get(personality, "Hello! I'm Sanniva.")
    with st.chat_message("assistant", avatar=get_avatar()):
        stream_data_to_chat(greeting)
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.greeting_shown = True


def main() -> None:
    st.set_page_config(page_title="Sanniva AI", page_icon="🤖")
    st.title("Chat With Sanniva!")
    st.sidebar.info(
        "I am Sanniva's Digital Twin! I can help with anything — and yeah, I'll tease you when you ask for it."
    )

    # Surface init errors so missing API keys are obvious instead of
    # silently failing on first chat.
    for err in _client_init_errors:
        st.sidebar.error(f"⚠️ {err}")

    initialize_session_state()
    _show_os_greeting()

    # --- Sidebar ---
    _sidebar_identity()
    _maybe_show_name_popup()
    personality, brain_type = _sidebar_personality_and_brain()
    temperature_val = _sidebar_model_settings()
    engine_label, voice, lang = _sidebar_tts()

    # --- Main panel ---
    display_chat_history()
    # Optimistic-update banner: shows "🔧 running tool…" the moment a tool
    # dispatch starts, even if the LLM stream is still going. Lives in its
    # own fragment so it can update without redrawing the chat history.
    _render_tool_status_banner()
    _show_initial_greeting(personality)

    if prompt := st.chat_input(get_catchy_phrase()):
        st.session_state.audio_bytes = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_prompt = build_system_prompt(
            # Pass the active persona so load_system_prompt strips the 6
            # inactive mode subsections at the source (~1,300-token saving
            # per request). Cached per-persona, so each mode is built once.
            load_system_prompt(personality),
            personality,
            brain_type,
            user_name=st.session_state.get("user_name", ""),
        )

        # Fast brain streams; Thinker brain still returns a finished string.
        use_stream = brain_type == "Fast"
        # Pick the whimsical "thinking" caption now so it stays consistent
        # for the lifetime of this turn (don't reroll each rerun).
        generating_phrase = pick_generating_phrase(personality)
        # Use a built-in spinner only for the network round-trip itself —
        # the in-bubble animated indicator handles the streaming phase.
        with st.spinner(f"🌀 {generating_phrase}"):
            response = get_ai_response_with_brain(
                prompt,
                system_prompt,
                brain_type,
                st.session_state.messages,
                temperature_val,
                stream=use_stream,
            )
        display_and_store_response(response, generating_phrase=generating_phrase)
        # Render any lore-save confirmations the tool layer queued during
        # this turn. They show up inline in the chat as compact captions —
        # NOT as popups or toasts — and are persisted into the transcript.
        _flush_lore_confirmations()
        # If a tool flipped the popup flag (e.g. request_user_name), rerun
        # so _maybe_show_name_popup() at the top of main() actually fires
        # it. Guard: only rerun if name is still unknown — otherwise we'd
        # loop when the popup already mutated the flag back to False.
        if (
            st.session_state.get("_name_popup_pending")
            and not (st.session_state.get("user_name") or "").strip()
        ):
            st.rerun()

    # --- Auto-play + manual speak controls for the latest assistant turn ---
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "assistant":
        _maybe_autoplay_response(engine_label, voice, lang)
        _handle_speak_button(msgs[-1]["content"], engine_label, voice, lang)


if __name__ == "__main__":
    main()
