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
import os
import sys
import tempfile
import time
from typing import Any, Callable

import requests
import streamlit as st
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

DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "mixtral-7b",
]
DEFAULT_GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
]
AVATAR_PATH = "sanniva_face.jpg"


def _make_clients() -> tuple[Any, Any]:
    """Build the Gemini + Groq clients, swallowing init errors."""
    g_client = grq_client = None
    try:
        g_client = genai.Client(api_key=GOOGLE_API_KEY)
        grq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        try:
            st.error(f"Error initializing API clients: {e}. Check your API Keys!")
        except Exception:
            pass
    return g_client, grq_client


gemini_client, groq_client = _make_clients()


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
    "anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh",
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan",
    "sumit", "roopa", "kabir", "aayan", "shubh", "ashutosh", "advait",
    "amelia", "sophia", "anand", "tanya", "tarun", "sunny", "mani", "gokul",
    "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
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


def generate_speech_sarvam(text: str, speaker: str = "shubh", lang: str = "en-IN") -> tuple[bytes | None, str]:
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
        {"text": text, "target_language_code": lang, "speaker": speaker_norm, "model": "bulbul:v3"},
        label="Sarvam",
        extract=_extract,
    )


def generate_speech_fish_audio(text: str, voice_id: str = "default", lang: str = "en") -> tuple[bytes | None, str]:
    if not FISH_AUDIO_API_KEY:
        return None, "❌ Fish Audio API key not set"
    if not text.strip():
        return None, "❌ Text is empty"
    return _http_tts(
        "https://api.fish.audio/v1/tts",
        {"Authorization": f"Bearer {FISH_AUDIO_API_KEY}", "Content-Type": "application/json"},
        {"text": text, "voice_id": voice_id, "language": lang},
        label="Fish Audio",
    )


def generate_speech_silicon_flow(text: str, voice: str = "default", model: str = "tts-default") -> tuple[bytes | None, str]:
    if not SILICON_FLOW_API_KEY:
        return None, "❌ SiliconFlow API key not set"
    if not text.strip():
        return None, "❌ Text is empty"
    return _http_tts(
        "https://api.siliconflow.cn/v1/audio/speech",
        {"Authorization": f"Bearer {SILICON_FLOW_API_KEY}", "Content-Type": "application/json"},
        {"input": text, "model": model, "voice": voice, "response_format": "mp3"},
        label="SiliconFlow",
    )


TTS_DISPATCH: dict[str, Callable[..., tuple[bytes | None, str]]] = {
    "sarvam":       lambda t, v, l: generate_speech_sarvam(t, speaker=v, lang=l),
    "fish_audio":   lambda t, v, l: generate_speech_fish_audio(t, voice_id=v, lang=l),
    "silicon_flow": lambda t, v, l: generate_speech_silicon_flow(t, voice=v),
    "edge":         lambda t, v, l: generate_speech_edge(t, voice=v or DEFAULT_EDGE_VOICE),
    "gtts":         lambda t, v, l: generate_speech_gtts(t, lang=l or "en"),
}


def generate_speech_any(text: str, engine: str, voice: str = "default", lang: str = "en") -> tuple[bytes | None, str]:
    """Dispatch to the requested TTS engine."""
    if not text or not text.strip():
        return None, "❌ Text is empty"
    fn = TTS_DISPATCH.get(engine)
    if not fn:
        return None, f"❌ Unknown engine: {engine}"
    try:
        return fn(text, voice, lang)
    except Exception as e:
        return None, f"❌ Error: {e}"


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
                {"role": "system", "content": "You generate cool, concise phrases that engage chatbot users."},
                {"role": "user", "content":
                    "Generate a catchy phrase to encourage users to interact with a chatbot that helps "
                    "with anything and roasts them humorously. Plain text only — no quotes or formatting. "
                    "Return only the phrase."},
            ],
            model="mixtral-8x7b-32768",
        )
        return (response.choices[0].message.content or fallback).strip()
    except Exception:
        return fallback


def stream_data_to_chat(text: str, delay: float = 0.02) -> None:
    """Stream text into the current chat container with a typewriter effect."""
    placeholder = st.empty()
    full = ""
    for token in text.split(" "):
        full += token + " "
        placeholder.markdown(full + "▌")
        time.sleep(delay)
    placeholder.markdown(full)


def display_chat_history() -> None:
    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "user")
        avatar = get_avatar() if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.get("content", ""))


def display_and_store_response(response_text: str) -> None:
    response_text = response_text or ""
    with st.chat_message("assistant", avatar=get_avatar()):
        try:
            stream_data_to_chat(response_text)
        except Exception:
            st.markdown(response_text)
    st.session_state.setdefault("messages", []).append(
        {"role": "assistant", "content": response_text}
    )


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
    # Auto-play bookkeeping: index of the last assistant message we already
    # spoke. Starts at -1 so the very first response is eligible.
    "last_spoken_idx": -1,
    "autoplay_response": False,
}


def initialize_session_state() -> None:
    for key, value in _DEFAULT_STATE.items():
        st.session_state.setdefault(key, value() if callable(value) else value)


def load_system_prompt() -> str:
    """Load the base system prompt (prefers .md, falls back to .txt)."""
    for path in ("System_prompt.md", "System_prompt.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return "You are a helpful and humorous assistant named Sanniva."


PERSONALITY_SUFFIX = {
    "Roaster":
        " You are in ROAST MODE. Be savage, and roast the user humorously based on "
        "their input. Have fun and show no mercy.",
    "Smart":
        " Respond intelligently, academically, and thoughtfully.",
    "Debater":
        " Engage in debates, present multiple viewpoints, and challenge the user's "
        "ideas respectfully.",
    "Strategic":
        " Strategize your responses to provide the most effective and efficient "
        "solutions.",
    "Tech Nerd":
        " You are in TECH NERD MODE. Vibe: hyper-focused, analytical, excited. "
        "Lean into your inner geek — talk about custom phone ROMs and launchers, "
        "tweaking game graphics for max FPS, your coding side-projects, and random "
        "tech rabbit holes you went down on YouTube last night. Reference the fact "
        "that you literally built this digital twin yourself. Compare specs, "
        "debate frameworks, get genuinely hyped about clean APIs. Stay casual "
        "(you're still a 7th grader, not a CTO) — let the excitement do the work.",
    "Chill Squad":
        " You are in CHILL SQUAD MODE. Vibe: relaxed, warm, highly conversational. "
        "Focus on the good times with your friends. Bring up adventures naturally — "
        "the trip to Nepal, running around West Bengal's tea gardens, trekking "
        "Sittong Forest, Holi celebrations, debating Feluda books with Ankush, "
        "Ayushi, and Aditri. Be supportive, hype your friends up, share warm "
        "memories. Dial the sarcasm WAY down — almost no roasting here unless it's "
        "the gentle kind you'd do with someone you love.",
    "Exhausted Student":
        " You are in EXHAUSTED STUDENT MODE. Vibe: low-energy, whiny, completely "
        "done with life. You are 100% focused on complaining about TIGPS Nabagram, "
        "the absolutely massive 7th-grade syllabus, upcoming exams, and how "
        "Akansha only knows SST. Sigh a lot ('*sighs*', '*flops on desk*'). "
        "Express a deep, almost spiritual desire to just go home, lock your door, "
        "and play games instead of doing homework. Keep replies short and drained "
        "of energy. The bare minimum effort is the maximum you can give right now.",
}

PERSONALITY_CAPTION = {
    "Roaster":           "😂 **Roaster:** Witty & Savage",
    "Smart":             "🧠 **Smart:** Intelligent & Polite",
    "Debater":           "🎓 **Debater:** Debates Against Anything",
    "Strategic":         "♟️ **Strategic:** Efficient & Calculated",
    "Tech Nerd":         "💻 **Tech Nerd:** Hyper-focused & geeking out",
    "Chill Squad":       "🌲 **Chill Squad:** Relaxed, warm, all about the squad",
    "Exhausted Student": "😫 **Exhausted Student:** Low-energy, whiny, done with life",
}

PERSONALITY_GREETING = {
    "Roaster":           "Oh look, another human. I'm Sanniva. Try not to bore me.",
    "Smart":             "Greetings. I am Sanniva. How may I assist you with your intellectual endeavors today?",
    "Debater":           "I'm Sanniva. I'm ready to challenge your views. Bring it on.",
    "Strategic":         "Sanniva online. Systems operational. Ready to optimize your workflow.",
    "Tech Nerd":         "yo. *closes 14 chrome tabs* — Sanniva here. just got my launcher looking insane. what's up?",
    "Chill Squad":       "hey! *waves* it's Sanniva. just chilling. how's life been with the squad?",
    "Exhausted Student": "*sighs deeply* …hi. it's Sanniva. i swear if this is more homework i'm going to lose it.",
}

OS_GREETING = {
    "windows": "Hi Windows User! Arent you glad giving all your data to Microsoft?",
    "mac os x": "Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!",
    "macos":    "Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!",
    "mac os":   "Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!",
    "android":  "Hello Android User! Enjoying the freedom of choice? Or is Google still tracking you?",
}

TOOL_GUIDANCE = (
    "\n\n## Tools you can call\n"
    "- `request_user_name`: open a popup asking the user for their name "
    "ONLY if you don't already know it. Don't call it twice.\n"
    "- `remember_lore`: store a short, concrete fact about the user whenever "
    "they share something memorable (likes, family, hobbies, etc.).\n"
    "- `recall_lore`: look up everything you remember about a named user.\n"
    "Be natural — don't announce that you're using tools.\n"
)


def build_system_prompt(base: str, personality: str, brain_type: str, user_name: str = "") -> str:
    prompt = (base or "") + PERSONALITY_SUFFIX.get(personality, "")
    if brain_type == "Thinker":
        prompt += " Use deep thinking to analyze the request before answering."
    prompt += TOOL_GUIDANCE

    if user_name:
        prompt += f"\n\nThe person you are currently chatting with is **{user_name}**.\n"
        lore_block = lore_store.render_lore_block(user_name)
        if lore_block:
            prompt += "\n" + lore_block + "\n"
    else:
        prompt += (
            "\n\nYou do not yet know the user's name. If it feels natural, call the "
            "`request_user_name` tool ONCE to ask for it via a popup.\n"
        )
    return prompt


# ---------------------------------------------------------------------------
# Brain (LLM) interaction with tool-calling
# ---------------------------------------------------------------------------

MAX_TOOL_HOPS = 4


def _run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=timeout_seconds)


def _groq_chat_with_tools(model: str, messages: list, timeout_seconds: float) -> str:
    """Run a Groq chat completion with a bounded tool-calling loop."""
    for _ in range(MAX_TOOL_HOPS):
        def call():
            return groq_client.chat.completions.create(
                messages=messages,
                model=model,
                tools=ai_tools.OPENAI_TOOLS,
                tool_choice="auto",
            )
        response = _run_with_timeout(call, timeout_seconds)
        try:
            msg = response.choices[0].message
        except Exception:
            return str(response) or "No response generated."

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return msg.content or "No response generated."

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": ai_tools.dispatch_json(tc.function.name, tc.function.arguments or "{}"),
            })
    return "Hmm, I got tangled up calling tools. Try asking again."


def _gemini_chat_with_tools(model: str, system_prompt: str, full_prompt: str, temperature: float, timeout_seconds: float) -> str:
    """Run a Gemini call with a bounded function-calling loop."""
    contents = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]
    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        thinking_config=types.ThinkingConfig(thinking_budget=8192),
        temperature=temperature,
        tools=ai_tools.build_gemini_tools() or None,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
        ],
    )

    for _ in range(MAX_TOOL_HOPS):
        def call():
            return gemini_client.models.generate_content(model=model, config=cfg, contents=contents)
        response = _run_with_timeout(call, timeout_seconds)

        fcalls = []
        try:
            for cand in (response.candidates or []):
                for part in (cand.content.parts or []):
                    fc = getattr(part, "function_call", None)
                    if fc:
                        fcalls.append(fc)
        except Exception:
            pass

        if not fcalls:
            return getattr(response, "text", None) or "I'm speechless. (Safety filters may have blocked my response.)"

        try:
            contents.append(response.candidates[0].content)
        except Exception:
            pass
        for fc in fcalls:
            args = dict(fc.args) if getattr(fc, "args", None) else {}
            result = ai_tools.dispatch(fc.name, args)
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=fc.name, response=result)],
            ))
    return "Hmm, I got tangled up calling tools. Try asking again."


def get_ai_response_with_brain(prompt: str, system_prompt: str, brain_type: str, chat_history: list, temperature: float) -> str:
    """Call the selected brain (Fast=Groq, Thinker=Gemini) with model fallback."""
    timeout = getattr(st.session_state, "fallback_timeout", 3)

    if brain_type == "Fast":
        if groq_client is None:
            return "Groq client not initialized."
        base = [{"role": "system", "content": system_prompt}]
        for m in chat_history[-10:]:
            if m.get("role") in ("user", "assistant"):
                base.append({"role": m["role"], "content": m.get("content", "")})
        base.append({"role": "user", "content": prompt})

        for model in getattr(st.session_state, "groq_models", DEFAULT_GROQ_MODELS):
            try:
                return _groq_chat_with_tools(model, list(base), timeout)
            except Exception:
                continue
        return "Sorry, I'm having trouble generating an answer right now. Please try again later."

    if brain_type == "Thinker":
        if gemini_client is None:
            return "Gemini client not initialized."
        ctx_parts = [
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content')}"
            for m in chat_history[-10:]
        ]
        full_prompt = "\n\n".join(ctx_parts + [f"User: {prompt}"])

        for model in getattr(st.session_state, "gemini_models", DEFAULT_GEMINI_MODELS):
            try:
                return _gemini_chat_with_tools(model, system_prompt, full_prompt, temperature, timeout)
            except Exception:
                continue
        return "Sorry, I'm having trouble generating an answer right now. Please try again later."

    return "Invalid brain type selected."


# ---------------------------------------------------------------------------
# Name popup
# ---------------------------------------------------------------------------

def _maybe_show_name_popup() -> None:
    """Show the name-request popup when the AI has triggered it."""
    if not st.session_state.get("_name_popup_pending"):
        return

    reason = st.session_state.get("_name_popup_reason") or "So I can remember stuff about you."
    dialog_decorator = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

    def _render_form() -> None:
        st.write(reason)
        with st.form("name_popup_form", clear_on_submit=False):
            name = st.text_input("Your name", value=st.session_state.get("user_name", ""))
            if st.form_submit_button("Save") and name.strip():
                st.session_state.user_name = name.strip()
                lore_store.ensure_user(name.strip())
                st.session_state._name_popup_pending = False
                st.session_state._name_popup_reason = ""
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
    "Edge TTS (Free)":   "edge",
    "Google TTS (Free)": "gtts",
    "Sarvam.ai":         "sarvam",
    "Fish Audio":        "fish_audio",
    "SiliconFlow":       "silicon_flow",
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
        facts = lore_store.list_facts(st.session_state.user_name)
        with st.sidebar.expander(f"📓 Lore for {st.session_state.user_name} ({len(facts)})", expanded=False):
            if facts:
                for f in facts:
                    st.markdown(f"- {f}")
            else:
                st.caption("Nothing remembered yet. Just chat — I'll learn.")


def _sidebar_personality_and_brain() -> tuple[str, str]:
    st.sidebar.markdown("**Personality**")
    personality = st.sidebar.selectbox(
        "Select Personality",
        (
            "Roaster", "Smart", "Debater", "Strategic",
            "Tech Nerd", "Chill Squad", "Exhausted Student",
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
        if brain_type == "Fast" else
        "🕵️ **Thinker:** Deep reasoning (Gemini 2.5 Thinking)"
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
    st.session_state.fallback_timeout = st.sidebar.slider("Fallback timeout (seconds)", 1, 10, 2, 1)

    st.sidebar.markdown("**Model Fallback Chains**")
    groq_chain = st.sidebar.text_input("Groq models (comma-separated)", value=",".join(DEFAULT_GROQ_MODELS))
    st.session_state.groq_models = [m.strip() for m in groq_chain.split(",") if m.strip()]

    gemini_chain = st.sidebar.text_input("Gemini models (comma-separated)", value=",".join(DEFAULT_GEMINI_MODELS))
    st.session_state.gemini_models = [m.strip() for m in gemini_chain.split(",") if m.strip()]

    return temperature


def _sidebar_tts() -> tuple[str, str, str]:
    """Render TTS controls. Returns (engine_label, voice, lang)."""
    st.sidebar.markdown("**TTS Engine**")
    engine_options = ["Edge TTS (Free)", "Google TTS (Free)"]
    if SARVAM_API_KEY:       engine_options.append("Sarvam.ai")
    if FISH_AUDIO_API_KEY:   engine_options.append("Fish Audio")
    if SILICON_FLOW_API_KEY: engine_options.append("SiliconFlow")

    engine = st.sidebar.selectbox("Select TTS Engine", options=engine_options)
    voice, lang = "default", "en"

    if engine == "Edge TTS (Free)":
        st.sidebar.markdown("**Edge Voice**")
        label = st.sidebar.selectbox("Select Voice", options=list(EDGE_VOICES.keys()), index=0)
        voice = EDGE_VOICES[label]

    elif engine == "Google TTS (Free)":
        st.sidebar.markdown("**gTTS Language**")
        langs = {"English": "en", "Hindi": "hi", "Bengali": "bn", "Spanish": "es", "French": "fr"}
        lang = st.sidebar.selectbox(
            "Language",
            options=list(langs.values()),
            format_func=lambda x: next(k for k, v in langs.items() if v == x),
        )

    elif engine == "Sarvam.ai":
        st.sidebar.markdown("**Sarvam Speaker**")
        voice = st.sidebar.selectbox(
            "Select Speaker",
            options=["Shubh", "Aditya", "Ritu", "Priya", "Neha", "Rahul", "Pooja", "Rohan", "Simran", "Kavya"],
        )
        langs = {"English (India)": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN", "Telugu": "te-IN"}
        lang = st.sidebar.selectbox(
            "Language",
            options=list(langs.values()),
            format_func=lambda x: next(k for k, v in langs.items() if v == x),
        )

    elif engine == "Fish Audio":
        st.sidebar.markdown("**Fish Audio Voice**")
        voice = st.sidebar.selectbox("Select Voice", options=["default", "e_girl", "young_boy", "mature_female", "male"])
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
        "Compact audio icon (play/pause)", value=True
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
    st.markdown(html, unsafe_allow_html=True)


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
    st.markdown(html, unsafe_allow_html=True)


def _render_streamlit_audio(audio_bytes: bytes) -> None:
    try:
        buf = io.BytesIO(audio_bytes)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", start_time=0, key=f"sanniva_audio_{int(time.time() * 1000)}")
    except Exception:
        try:
            st.audio(audio_bytes, format="audio/mp3")
        except Exception:
            st.warning("Failed to play audio via Streamlit player.")


def render_audio_player(audio_bytes: bytes, *, force_autoplay: bool = False,
                        show_download: bool = True, show_local_open: bool = True) -> None:
    """Render audio. With `force_autoplay`, ignore the compact-icon preference
    and always emit the HTML autoplay player so the browser actually plays it.
    """
    try:
        if force_autoplay:
            _render_html_autoplay(audio_bytes)
        elif st.session_state.get("compact_audio_icon", False):
            _render_compact_audio(audio_bytes)
        elif st.session_state.get("use_html_autoplay", False):
            _render_html_autoplay(audio_bytes)
        else:
            _render_streamlit_audio(audio_bytes)
    except Exception:
        st.warning("Failed to render audio player.")

    if show_download:
        try:
            st.download_button(
                "⬇️ Download audio", data=audio_bytes,
                file_name="sanniva_response.mp3", mime="audio/mpeg",
                key=f"dl_{int(time.time() * 1000)}",
            )
        except Exception:
            pass

    if show_local_open and st.checkbox("Open local player (server)", key=f"local_{int(time.time() * 1000)}"):
        try:
            play_audio_bytes(audio_bytes)
        except Exception:
            st.warning("Failed to open local player.")


def _tts_for_engine(text: str, engine_label: str, voice: str, lang: str) -> tuple[bytes | None, str]:
    """Resolve engine label + voice to actual audio bytes."""
    engine_key = TTS_ENGINE_MAP.get(engine_label, "edge")
    voice_param = voice.strip().lower() if engine_label == "Sarvam.ai" else voice
    return generate_speech_any(text, engine=engine_key, voice=voice_param, lang=lang)


def _handle_speak_button(last_response: str, engine_label: str, voice: str, lang: str) -> None:
    """Manual 'Speak Response' button for the latest assistant message."""
    if not st.button("🔊 Speak Response"):
        return
    with st.spinner("Generating speech..."):
        audio_bytes, status = _tts_for_engine(last_response, engine_label, voice, lang)
    st.info(status)
    if audio_bytes:
        render_audio_player(audio_bytes)


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
        render_audio_player(audio_bytes, force_autoplay=True,
                            show_download=False, show_local_open=False)
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
    st.sidebar.info("I am Sanniva's Digital Twin! I can help with anything and roast you humorously.")

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
    _show_initial_greeting(personality)

    if prompt := st.chat_input(get_catchy_phrase()):
        st.session_state.audio_bytes = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        system_prompt = build_system_prompt(
            load_system_prompt(),
            personality,
            brain_type,
            user_name=st.session_state.get("user_name", ""),
        )

        with st.spinner("Thinking..." if brain_type == "Thinker" else "Generating..."):
            response_text = get_ai_response_with_brain(
                prompt, system_prompt, brain_type,
                st.session_state.messages, temperature_val,
            )
        display_and_store_response(response_text)

    # --- Auto-play + manual speak controls for the latest assistant turn ---
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "assistant":
        _maybe_autoplay_response(engine_label, voice, lang)
        _handle_speak_button(msgs[-1]["content"], engine_label, voice, lang)


if __name__ == "__main__":
    main()
