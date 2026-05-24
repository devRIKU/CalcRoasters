"""
Free TTS providers
==================
Wraps free, no-API-key TTS engines so the chatbot has a zero-cost path
even when paid keys (Sarvam, Fish, SiliconFlow) are missing.

Engines:
- edge   : Microsoft Edge's online TTS (via `edge-tts` pkg). Very high
           quality, many voices, completely free, no key required.
- gtts   : Google Translate TTS (via `gTTS` pkg). Free, simple, robust.

Both return (audio_bytes_mp3, status_message).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import sys
from typing import Optional


# A small curated list of good free Edge voices.
EDGE_VOICES = {
    "Ava (US, female)":      "en-US-AvaNeural",
    "Andrew (US, male)":     "en-US-AndrewNeural",
    "Emma (US, female)":     "en-US-EmmaNeural",
    "Brian (US, male)":      "en-US-BrianNeural",
    "Aria (US, female)":     "en-US-AriaNeural",
    "Guy (US, male)":        "en-US-GuyNeural",
    "Jenny (US, female)":    "en-US-JennyNeural",
    "Sonia (UK, female)":    "en-GB-SoniaNeural",
    "Ryan (UK, male)":       "en-GB-RyanNeural",
    "Neerja (India, female)":"en-IN-NeerjaNeural",
    "Prabhat (India, male)": "en-IN-PrabhatNeural",
    "Swara (Hindi, female)": "hi-IN-SwaraNeural",
    "Madhur (Hindi, male)":  "hi-IN-MadhurNeural",
}

DEFAULT_EDGE_VOICE = "en-US-AvaNeural"


def _run_edge_in_dedicated_loop(coro_factory, timeout: float = 60.0) -> bytes:
    """Run an edge-tts coroutine on a brand-new event loop in a dedicated
    worker thread, and return the resulting bytes.

    Why this gymnastics:
      * Streamlit's script-runner is NOT the main thread, so naive
        `asyncio.run()` from a Streamlit callback can hit
        "RuntimeError: There is no current event loop in thread 'X'" or pick
        up a stale, *closed* loop attached by a previous rerun.
      * `asyncio.get_event_loop()` is deprecated and silently surprising on
        3.10+ — it can return a closed loop on a worker thread without
        raising, so `run_until_complete` then crashes with "Event loop is
        closed". The previous fallback chain caught the RuntimeError but
        could still end up with an empty BytesIO from aiohttp tearing down
        mid-stream, hence the long-standing "blank audio file" symptom.
      * On Windows we explicitly force the SelectorEventLoop — the default
        ProactorEventLoop on non-main threads has known cleanup issues with
        aiohttp (which edge-tts uses under the hood) and is the most likely
        culprit for empty-but-no-error audio.

    Doing all of this on a *dedicated* thread also guarantees that
    `aiohttp`'s connector / TCPSocket cleanup completes before we read the
    bytes out — no race with the loop being closed mid-teardown.
    """

    def _runner() -> bytes:
        # Force a Selector loop on Windows; Proactor + aiohttp on a non-main
        # thread is the source of the empty-audio-no-exception failures.
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                # Older Pythons or restricted envs may not allow this; the
                # SelectorEventLoop we create below is what really matters.
                pass

        try:
            loop = asyncio.SelectorEventLoop()
        except Exception:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            # Drain any remaining tasks before closing so aiohttp's
            # connector teardown actually completes — this is what stops
            # the buffer from being silently empty.
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_runner).result(timeout=timeout)


def generate_speech_edge(
    text: str,
    voice: str = DEFAULT_EDGE_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> tuple[Optional[bytes], str]:
    """Generate speech via Microsoft Edge TTS (free, no API key).

    Returns `(audio_mp3_bytes_or_None, status_message)`. The status message
    is always populated — when it starts with ``❌`` the caller should
    surface it (e.g. via `st.error`) so blank audio doesn't fail silently.
    """
    if not text or not text.strip():
        return None, "❌ Text is empty"
    try:
        import edge_tts  # type: ignore
    except ImportError:
        return None, "❌ edge-tts not installed. Run: pip install edge-tts"

    async def _run() -> bytes:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    try:
        audio = _run_edge_in_dedicated_loop(_run, timeout=60.0)
    except concurrent.futures.TimeoutError:
        return None, "❌ Edge TTS timed out after 60s (network or Bing TTS backend)"
    except Exception as e:
        # edge-tts raises edge_tts.exceptions.NoAudioReceived for bad voice IDs.
        return None, f"❌ Edge TTS error: {type(e).__name__}: {e}"

    if not audio:
        # The async stream finished cleanly but produced no audio bytes.
        # This almost always means the voice ID is wrong, the text was
        # filtered by Bing TTS, or the Bing endpoint returned an empty
        # stream. Report it instead of returning an empty file.
        return None, (
            f"❌ Edge TTS returned 0 bytes (voice='{voice}'). "
            "Check the voice ID against `edge-tts --list-voices`, or try a different voice."
        )
    return audio, f"✅ Edge TTS generated {len(audio)} bytes ({voice})"


def generate_speech_gtts(text: str, lang: str = "en", tld: str = "com") -> tuple[Optional[bytes], str]:
    """Generate speech via Google Translate TTS (free, no API key)."""
    if not text or not text.strip():
        return None, "❌ Text is empty"
    try:
        from gtts import gTTS  # type: ignore
    except ImportError:
        return None, "❌ gTTS not installed. Run: pip install gTTS"

    try:
        tts = gTTS(text=text, lang=lang, tld=tld)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio = buf.getvalue()
        return audio, f"✅ gTTS generated {len(audio)} bytes ({lang})"
    except Exception as e:
        return None, f"❌ gTTS error: {e}"
