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
import io
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


def generate_speech_edge(
    text: str,
    voice: str = DEFAULT_EDGE_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> tuple[Optional[bytes], str]:
    """Generate speech via Microsoft Edge TTS (free, no API key)."""
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
        # Run async function — handle the case where an event loop already exists
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use a new thread loop
                import concurrent.futures

                def _runner() -> bytes:
                    new_loop = asyncio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(_run())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    audio = ex.submit(_runner).result(timeout=60)
            else:
                audio = loop.run_until_complete(_run())
        except RuntimeError:
            audio = asyncio.run(_run())

        if not audio:
            return None, "❌ Edge TTS returned empty audio"
        return audio, f"✅ Edge TTS generated {len(audio)} bytes ({voice})"
    except Exception as e:
        return None, f"❌ Edge TTS error: {e}"


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
