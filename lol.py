from dotenv import load_dotenv
import os
import sys
import tempfile
import time

load_dotenv()

API_KEY = "sk_85e48c116e25e8c9ec047a60bee7104d6a89421ca3111cf0"
if not API_KEY:
  print("Error: ELEVENLABS_API_KEY is not set in environment. Set it in .env or environment variables.")
  sys.exit(1)

# Try to import common elevenlabs SDK entry points (works across versions)
ElevenLabs = None
play_fn = None
try:
  # newer packaging may expose top-level helpers
  from elevenlabs import ElevenLabs as _ElevenLabs, play as _play
  ElevenLabs = _ElevenLabs
  play_fn = _play
except Exception:
  try:
    # older packaging path
    from elevenlabs.client import ElevenLabs as _ElevenLabs
    ElevenLabs = _ElevenLabs
  except Exception:
    ElevenLabs = None

client = None
if ElevenLabs:
  try:
    client = ElevenLabs(api_key=API_KEY)
  except Exception as e:
    print(f"Warning: unable to create ElevenLabs client: {e}")

# Text and voice config
TEXT = "The first move is what sets everything in motion."
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"


def _extract_bytes(obj):
  """Try common ways to extract raw bytes from SDK response objects."""
  if obj is None:
    return None
  # raw bytes
  if isinstance(obj, (bytes, bytearray)):
    return bytes(obj)
  # file-like
  if hasattr(obj, "read"):
    try:
      return obj.read()
    except Exception:
      pass
  # some SDKs return attributes like 'content' or 'audio'
  for attr in ("content", "audio", "raw", "data"):
    if hasattr(obj, attr):
      val = getattr(obj, attr)
      if isinstance(val, (bytes, bytearray)):
        return bytes(val)
      if hasattr(val, "read"):
        try:
          return val.read()
        except Exception:
          pass
  # as a last resort, try str -> bytes
  try:
    s = str(obj)
    return s.encode("utf-8")
  except Exception:
    return None


def _save_temp_mp3(data: bytes) -> str:
  f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
  f.write(data)
  f.flush()
  f.close()
  return f.name


def main():
  audio_bytes = None

  # Prefer SDK client method if available
  if client is not None and hasattr(client, "text_to_speech"):
    try:
      # Many SDK versions provide a .convert() or .generate() method
      tts = client.text_to_speech
      if hasattr(tts, "convert"):
        resp = tts.convert(text=TEXT, voice_id=VOICE_ID, model_id=MODEL_ID, output_format="mp3_44100_128")
      elif hasattr(tts, "generate"):
        resp = tts.generate(text=TEXT, voice=VOICE_ID, model=MODEL_ID)
      else:
        # fallback: try calling as function
        resp = tts(text=TEXT, voice_id=VOICE_ID)
      audio_bytes = _extract_bytes(resp)
    except Exception as e:
      print(f"Error generating audio with client.text_to_speech: {e}")

  # If SDK top-level generate is available
  if audio_bytes is None:
    try:
      from elevenlabs import generate as eleven_generate
      resp = eleven_generate(text=TEXT, voice=VOICE_ID, model=MODEL_ID)
      audio_bytes = _extract_bytes(resp)
    except Exception:
      pass

  # If still none, error out
  if not audio_bytes:
    print("Failed to generate audio using ElevenLabs SDK. Check your SDK version and API key.")
    sys.exit(1)

  # Save and play
  mp3_path = _save_temp_mp3(audio_bytes)
  print(f"Saved audio to: {mp3_path}")

  # Try SDK play function if available (many implementations accept bytes)
  if play_fn is not None:
    try:
      # prefer passing raw bytes
      play_fn(audio_bytes)
      return
    except Exception:
      try:
        # some variants accept a filename
        play_fn(mp3_path)
        return
      except Exception:
        pass

  # Fallback: on Windows, use os.startfile; on other platforms, try 'open' or 'xdg-open'
  try:
    if sys.platform.startswith("win"):
      os.startfile(mp3_path)  # type: ignore
    elif sys.platform == "darwin":
      os.system(f"open {mp3_path}")
    else:
      os.system(f"xdg-open {mp3_path}")
  except Exception as e:
    print(f"Could not open audio file automatically: {e}. File saved at: {mp3_path}")


if __name__ == "__main__":
  main()


