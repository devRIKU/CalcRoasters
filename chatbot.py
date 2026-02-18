from google import genai
from google.genai import types
import streamlit as st
import os
import time
import sys
import tempfile
from groq import Groq
from streamlit.runtime.scriptrunner import get_script_run_ctx
import io
from user_agents import parse  # You will need to install this!
import requests
import base64
import concurrent.futures

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- CONFIGURATION ---
# Make sure these are set in your environment or Streamlit Secrets!
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# TTS SERVICE API KEYS ---
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
FISH_AUDIO_API_KEY = os.environ.get("FISH_AUDIO_API_KEY")
SILICON_FLOW_API_KEY = os.environ.get("SILICON_FLOW_API_KEY")

# Initialize Clients with error handling
client = None
client2 = None
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    client2 = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    # Avoid raising at import time; show message once app runs
    try:
        st.error(f"Error initializing API clients: {e}. Check your API Keys!")
    except Exception:
        pass


def get_user_agent_string():
    """Gets the raw User-Agent string from the current session's headers."""
    try:
        # Get the current session context
        try:
            ctx = get_script_run_ctx()
        except RuntimeError:
            # Running outside Streamlit context (e.g., with `python chatbot.py`)
            return "Running outside Streamlit context"
        
        if ctx is None:
            return "Could not get session context."
        
        # Access the headers from the session's request
        # Note: This uses Streamlit's internal runtime, which is generally discouraged, 
        # but it's the most reliable way to get headers right now.
        try:
            headers = st.runtime.get_instance().get_client(ctx.session_id).request.headers
        except Exception:
            return "Could not get headers from session."
        
        # The User-Agent is the specific header we want
        user_agent_string = headers.get("User-Agent", "User-Agent Not Found")
        return user_agent_string
        
    except Exception as e:
        return f"Error retrieving User-Agent: {e}"


def generate_speech_sarvam(text: str, speaker: str = "Shubh", lang: str = "en-IN") -> tuple[bytes | None, str]:
    """Generate speech using Sarvam.ai TTS. Returns (audio_bytes, error_msg)."""
    if not SARVAM_API_KEY:
        return None, "❌ Sarvam API key not set"
    if not text or not text.strip():
        return None, "❌ Text is empty"
    try:
        # Normalize speaker name to the expected lowercase tokens used by the Sarvam API
        allowed_speakers = {"anushka","abhilash","manisha","vidya","arya","karun","hitesh","aditya","ritu","priya","neha","rahul","pooja","rohan","simran","kavya","amit","dev","ishita","shreya","ratan","varun","manan","sumit","roopa","kabir","aayan","shubh","ashutosh","advait","amelia","sophia","anand","tanya","tarun","sunny","mani","gokul","vijay","shruti","suhani","mohit","kavitha","rehan","soham","rupali"}
        speaker_norm = (speaker or "").strip().lower()
        fallback_note = ""
        if speaker_norm not in allowed_speakers:
            # fallback to a known default and inform the caller
            fallback_note = f"❗ Speaker '{speaker}' not supported by Sarvam; falling back to 'shubh'."
            speaker_norm = "shubh"
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "target_language_code": lang,
            "speaker": speaker_norm,
            "model": "bulbul:v3"
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "audios" in data and len(data["audios"]) > 0:
                audio_b64 = data["audios"][0]
                audio_bytes = base64.b64decode(audio_b64)
                note = f"✅ Generated {len(audio_bytes)} bytes"
                if fallback_note:
                    note = fallback_note + " " + note
                return audio_bytes, note
            return None, f"❌ No audio in response: {data}"
        return None, f"❌ API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"❌ Sarvam error: {str(e)}"


def generate_speech_fish_audio(text: str, voice_id: str = "default", lang: str = "en") -> tuple[bytes | None, str]:
    """Generate speech using Fish Audio TTS. Returns (audio_bytes, error_msg)."""
    if not FISH_AUDIO_API_KEY:
        return None, "❌ Fish Audio API key not set"
    if not text or not text.strip():
        return None, "❌ Text is empty"
    try:
        url = "https://api.fish.audio/v1/tts"
        headers = {
            "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": lang
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            audio_bytes = resp.content
            return audio_bytes, f"✅ Generated {len(audio_bytes)} bytes"
        return None, f"❌ API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"❌ Fish Audio error: {str(e)}"


def generate_speech_silicon_flow(text: str, model: str = "tts-default", voice: str = "default") -> tuple[bytes | None, str]:
    """Generate speech using SiliconFlow TTS. Returns (audio_bytes, error_msg)."""
    if not SILICON_FLOW_API_KEY:
        return None, "❌ SiliconFlow API key not set"
    if not text or not text.strip():
        return None, "❌ Text is empty"
    try:
        url = "https://api.siliconflow.cn/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {SILICON_FLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": "mp3"
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            audio_bytes = resp.content
            return audio_bytes, f"✅ Generated {len(audio_bytes)} bytes"
        return None, f"❌ API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"❌ SiliconFlow error: {str(e)}"


def generate_speech_elevenlabs(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", model_id: str = "eleven_multilingual_v2") -> tuple[bytes | None, str]:
    """Generate speech using ElevenLabs TTS. Returns (audio_bytes, error_msg)."""
    return None, "❌ ElevenLabs support removed"


def generate_speech_any(text: str, engine: str = "sarvam", speaker_or_voice: str = "Shubh", lang: str = "en") -> tuple[bytes | None, str]:
    """Generate speech using the selected engine. Returns (audio_bytes, error_msg)."""
    if not text or not text.strip():
        return None, "❌ Text is empty"
    try:
        if engine == "sarvam":
            return generate_speech_sarvam(text, speaker=speaker_or_voice, lang=lang)
        elif engine == "fish_audio":
            return generate_speech_fish_audio(text, voice_id=speaker_or_voice, lang=lang)
        elif engine == "silicon_flow":
            return generate_speech_silicon_flow(text, voice=speaker_or_voice)
        else:
            return None, f"❌ Unknown engine: {engine}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def play_audio_bytes(audio_bytes: bytes):
    """Play audio bytes using the default player on the system."""
    if not audio_bytes:
        return
    try:
        # Save to temp file
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        f.write(audio_bytes)
        f.flush()
        f.close()
        # Open with OS default player
        if sys.platform.startswith("win"):
            os.startfile(f.name)  # type: ignore
        elif sys.platform == "darwin":
            os.system(f"open {f.name}")
        else:
            os.system(f"xdg-open {f.name}")
    except Exception:
        pass

# --- Function to parse the OS from the User-Agent string ---
def get_os_from_user_agent(user_agent_string):
    """Uses the 'user-agents' library to parse OS info."""
    if not user_agent_string or "Not Found" in user_agent_string or "Error" in user_agent_string \
       or "Could not get session context." in user_agent_string or "Could not get headers" in user_agent_string \
       or "Running outside Streamlit context" in user_agent_string:
        return "Unknown OS"
    
    # Use the parsing library
    try:
        user_agent = parse(user_agent_string)
        # Get the OS information (e.g., 'Windows', 'Mac OS X', 'Android')
        return user_agent.os.family or "Unknown OS"
    except Exception:
        return "Unknown OS"

# Helper for Avatar
def get_avatar():
    """Returns image path if exists, else emoji"""
    try:
        if os.path.exists("sanniva_face.jpg"):
            return "sanniva_face.jpg"
    except Exception:
        pass
    return "🤖"  # Fallback emoji

@st.cache_data(ttl=3600)
def get_catchy_phrase() -> str:
    """Generate a catchy phrase using Groq."""
    if client2 is None:
        return "Yeah go ahead, ask me anything!"
    try:
        response = client2.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You generate cool and concise Phrases to engage users to my Chatbot application.",
                },
                {
                    "role": "user",
                    "content": "Generate a catchy phrase to encourage users to interact with a chatbot that helps with anything and roasts them humorously. Dont use any formatting like quotes or special characters and bold. Keep it short and sweet. Return only the phrase.",
                },
            ],
            model="mixtral-8x7b-32768",
        )
        # Defensive access
        try:
            return response.choices[0].message.content or "Yeah go ahead, ask me anything!"
        except Exception:
            return str(response) or "Yeah go ahead, ask me anything!"
    except Exception:
        return "Yeah go ahead, ask me anything!"

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # default personality selector value
    if "personality_selector" not in st.session_state:
        st.session_state.personality_selector = "Roaster"
    if "greeting_shown" not in st.session_state:
        st.session_state.greeting_shown = False
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

def display_chat_history():
    """Display all messages in the chat history."""
    if "messages" not in st.session_state:
        return
    for message in st.session_state.messages:
        role = message.get("role", "user")
        avatar = get_avatar() if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.get("content", ""))

def stream_data_to_chat(text: str, delay: float = 0.02):
    """Streams data with a typing effect."""
    placeholder = st.empty()
    full_response = ""
    tokens = text.split(" ")
    for token in tokens:
        full_response += token + " "
        placeholder.markdown(full_response + "▌")
        time.sleep(delay)
    placeholder.markdown(full_response)

def load_system_prompt() -> str:
    """Load the base system prompt from file."""
    try:
        with open("System_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful and humorous assistant named Sanniva."
    except Exception:
        return "You are a helpful and humorous assistant named Sanniva."

def get_ai_response_with_brain(prompt: str, system_prompt: str, brain_type: str, chat_history: list, temperature: float) -> str:
    """Get AI response based on selected brain type."""
    try:
        # helper to run a blocking call with a timeout using a thread
        def run_with_timeout(fn, timeout_seconds: float):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                try:
                    return fut.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError("API call timed out")

        if brain_type == "Fast":
            # --- GROQ LOGIC ---
            if client2 is None:
                return "Groq client not initialized."
            # fallback chain for Groq models (from session_state if present)
            groq_models = getattr(st.session_state, "groq_models", [
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "mixtral-7b"
            ])
            messages_base = [{"role": "system", "content": system_prompt}]
            for msg in chat_history[-10:]:
                messages_base.append({"role": msg.get("role"), "content": msg.get("content")})
            messages_base.append({"role": "user", "content": prompt})

            # default timeout (seconds) — can be overridden by caller via st.sidebar
            timeout_seconds = getattr(st.session_state, "fallback_timeout", 3)

            last_err = None
            for model in groq_models:
                try:
                    def call_groq():
                        return client2.chat.completions.create(messages=messages_base, model=model)

                    response = run_with_timeout(call_groq, timeout_seconds)
                    try:
                        return response.choices[0].message.content or "No response generated."
                    except Exception:
                        return str(response) or "No response generated."
                except Exception as e:
                    last_err = e
                    # try next model in the chain
                    continue
            # all Groq attempts failed
            return "Sorry, I'm having trouble generating an answer right now. Please try again later."

        elif brain_type == "Thinker":
            # --- GEMINI LOGIC ---
            if client is None:
                return "Gemini client not initialized."
            # conversation assembly
            conversation_context = ""
            for msg in chat_history[-10:]:
                role_label = "User" if msg.get("role") == "user" else "Assistant"
                conversation_context += f"{role_label}: {msg.get('content')}\n\n"
            full_prompt = f"{conversation_context}User: {prompt}"

            # model fallback chain for Gemini (from session_state if present)
            gemini_models = getattr(st.session_state, "gemini_models", [
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
                "gemini-2.5-lite"
            ])

            timeout_seconds = getattr(st.session_state, "fallback_timeout", 3)

            last_err = None
            for model in gemini_models:
                try:
                    def call_gemini():
                        return client.models.generate_content(
                            model=model,
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                thinking_config=types.ThinkingConfig(thinking_budget=8192),
                                temperature=temperature,
                                safety_settings=[
                                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH")
                                ]
                            ),
                            contents=[full_prompt],
                        )

                    response = run_with_timeout(call_gemini, timeout_seconds)
                    if getattr(response, "text", None):
                        return response.text
                    try:
                        return str(response)
                    except Exception:
                        return "I'm speechless. (Safety filters might have blocked my response)."
                except Exception as e:
                    last_err = e
                    # try next model in chain
                    continue

            # all Gemini attempts failed
            return "Sorry, I'm having trouble generating an answer right now. Please try again later."
        else:
            return "Invalid brain type selected."

    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_and_store_response(response_text: str):
    """Display AI response with streaming effect and store in session."""
    if response_text is None:
        response_text = ""
    with st.chat_message("assistant", avatar=get_avatar()):
        try:
            stream_data_to_chat(response_text)
        except Exception:
            st.markdown(response_text)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": response_text})

def build_system_prompt(base_prompt: str, personality: str, brain_type: str) -> str:
    prompt = base_prompt or ""
    if personality == "Roaster":
        prompt += " You are in ROAST MODE. Be savage, and roast the user humorously based on their input. Have Fun and no mercy"
    elif personality == "Smart":
        prompt += "Respond intelligently, academically, and thoughtfully."
    elif personality == "Debater":
        prompt += " Engage in debates, present multiple viewpoints, and challenge the user's ideas respectfully."
    elif personality == "Strategic":
        prompt += "Strategize your responses to provide the most effective and efficient solutions."
    if brain_type == "Thinker":
        prompt += " Use deep thinking to analyze the request before answering."
    return prompt


# Auto-personality detection removed — personality is now always manual via the sidebar selectbox

def main():
    raw_ua = get_user_agent_string()
    user_os = get_os_from_user_agent(raw_ua)
    user_os_lower = (user_os or "").lower()
    try:
        if user_os_lower == "windows":
            st.sidebar.success("Hi Windows User! Arent you glad giving all your data to Microsoft?")
        elif user_os_lower in ("mac os x", "macos", "mac os"):
            st.sidebar.success("Hey Mac User! Enjoying the walled garden? Hope you like paying for wheels!")
        elif user_os_lower == "android":
            st.sidebar.success("Hello Android User! Enjoying the freedom of choice? Or is Google still tracking you?")
    except Exception:
        pass

    st.set_page_config(page_title="Sanniva AI", page_icon="🤖")
    st.title("Chat With Sanniva!")
    st.sidebar.info("I am Sanniva's Digital Twin! I can help with anything and roast you humorously.")

    # Initialize session state early so detection and UI stay in sync
    initialize_session_state()

    # Personality selector (manual only)
    st.sidebar.markdown("**Personality**")
    personality = st.sidebar.selectbox(
        "Select Personality",
        ("Roaster", "Smart", "Debater", "Strategic"),
        key="personality_selector",
        label_visibility="collapsed",
    )

    # Apply Theme based on Personality (updates config.toml)
    from styles import apply_theme
    apply_theme(personality)

    if personality == "Roaster":
        st.sidebar.caption("😂 **Roaster:** Witty & Savage")
    elif personality == "Smart":
        st.sidebar.caption("🧠 **Smart:** Intelligent & Polite")
    elif personality == "Debater":
        st.sidebar.caption("🎓 **Debater:** Debates Against Anything")
    elif personality == "Strategic":
        st.sidebar.caption("♟️ **Strategic:** Efficient & Calculated")

    # Brain Selector
    st.sidebar.markdown("**Brain Power**")
    brain_type = st.sidebar.selectbox(
        "Select Brain",
        ("Fast", "Thinker"),
        key="brain_selector",
        label_visibility="collapsed"
    )

    if brain_type == "Fast":
        st.sidebar.caption("⚡ **Fast:** Instant answers (Groq)")
    else:
        st.sidebar.caption("🕵️ **Thinker:** Deep reasoning (Gemini 2.5 Thinking)")

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    temperature_val = st.sidebar.slider(
        "Creativity Level (Chaos)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    # Fallback timeout for trying alternate models (seconds)
    st.sidebar.markdown("**Fallback Settings**")
    fallback_timeout = st.sidebar.slider(
        "Fallback timeout (seconds)",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )
    st.session_state.fallback_timeout = fallback_timeout

    # Model fallback chains (comma-separated). User can customize order here.
    st.sidebar.markdown("**Model Fallback Chains**")
    default_groq = "llama-3.3-70b-versatile,mixtral-8x7b-32768,mixtral-7b"
    groq_chain_str = st.sidebar.text_input("Groq models (comma-separated)", value=default_groq)
    groq_chain = [m.strip() for m in groq_chain_str.split(",") if m.strip()]
    st.session_state.groq_models = groq_chain

    default_gemini = "gemini-3-flash-preview,gemini-2.5-flash,gemini-2.5-lite"
    gemini_chain_str = st.sidebar.text_input("Gemini models (comma-separated)", value=default_gemini)
    gemini_chain = [m.strip() for m in gemini_chain_str.split(",") if m.strip()]
    st.session_state.gemini_models = gemini_chain

    # --- TTS Engine & Voice Selector ---
    st.sidebar.markdown("**TTS Engine**")
    engine_options = []
    if SARVAM_API_KEY:
        engine_options.append("Sarvam.ai")
    if FISH_AUDIO_API_KEY:
        engine_options.append("Fish Audio")
    if SILICON_FLOW_API_KEY:
        engine_options.append("SiliconFlow")
    
    if not engine_options:
        st.sidebar.warning("No TTS engines configured. Add API keys to .env file.")
        selected_engine = "none"
    else:
        selected_engine = st.sidebar.selectbox("Select TTS Engine", options=engine_options)

    # Voice/Speaker selector based on engine
    selected_voice = "default"
    selected_lang = "en"
    
    if selected_engine == "Sarvam.ai":
        st.sidebar.markdown("**Sarvam Speaker**")
        sarvam_speakers = ["Shubh", "Aditya", "Ritu", "Priya", "Neha", "Rahul", "Pooja", "Rohan", "Simran", "Kavya"]
        selected_voice = st.sidebar.selectbox("Select Speaker", options=sarvam_speakers)
        sarvam_langs = {"English (India)": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN", "Telugu": "te-IN"}
        selected_lang = st.sidebar.selectbox("Language", options=list(sarvam_langs.values()), format_func=lambda x: [k for k, v in sarvam_langs.items() if v == x][0])
    
    elif selected_engine == "Fish Audio":
        st.sidebar.markdown("**Fish Audio Voice**")
        fish_voices = ["default", "e_girl", "young_boy", "mature_female", "male"]
        selected_voice = st.sidebar.selectbox("Select Voice", options=fish_voices)
        fish_langs = {"English": "en", "Chinese": "zh", "Spanish": "es", "French": "fr"}
        selected_lang = st.sidebar.selectbox("Language", options=list(fish_langs.values()), format_func=lambda x: [k for k, v in fish_langs.items() if v == x][0])
    
    elif selected_engine == "SiliconFlow":
        st.sidebar.markdown("**SiliconFlow Voice**")
        sf_voices = ["default", "narrator_en", "narrator_zh", "casual_en", "casual_zh"]
        selected_voice = st.sidebar.selectbox("Select Voice", options=sf_voices)
        selected_lang = "en"

    # (ElevenLabs support removed)
    display_chat_history()

    # Initial Greeting with Typewriter Effect
    if not st.session_state.greeting_shown:
        if personality == "Roaster":
            greeting_text = "Oh look, another human. I'm Sanniva. Try not to bore me."
        elif personality == "Smart":
            greeting_text = "Greetings. I am Sanniva. How may I assist you with your intellectual endeavors today?"
        elif personality == "Debater":
            greeting_text = "I'm Sanniva. I'm ready to challenge your views. Bring it on."
        elif personality == "Strategic":
            greeting_text = "Sanniva online. Systems operational. Ready to optimize your workflow."
        else:
            greeting_text = "Hello! I'm Sanniva."

        with st.chat_message("assistant", avatar=get_avatar()):
            stream_data_to_chat(greeting_text)
        
        st.session_state.messages.append({"role": "assistant", "content": greeting_text})
        st.session_state.greeting_shown = True

    # Retrieve user agent and OS info now that Streamlit context exists


    # Show OS-specific sidebar messages safely
   
    # (No auto-personality) — personality comes from the sidebar selectbox

    catchy_text = get_catchy_phrase()

    if prompt := st.chat_input(catchy_text):
        st.session_state.audio_bytes = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        base_system_prompt = load_system_prompt()
        system_prompt = build_system_prompt(base_system_prompt, personality, brain_type)

        with st.spinner("Thinking..." if brain_type == "Thinker" else "Generating..."):
            response_text = get_ai_response_with_brain(
                prompt,
                system_prompt,
                brain_type,
                st.session_state.messages,
                temperature_val
            )

        display_and_store_response(response_text)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_response = st.session_state.messages[-1]["content"]
        # Add a "Speak" button to read the response aloud
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔊 Speak Response"):
                with st.spinner("Generating speech..."):
                    # Map engine name to engine key
                    engine_map = {
                        "Sarvam.ai": "sarvam",
                        "Fish Audio": "fish_audio",
                        "SiliconFlow": "silicon_flow",
                    }
                    engine_key = engine_map.get(selected_engine, "sarvam")
                    # Normalize Sarvam speaker names to lowercase tokens expected by the API
                    speaker_param = selected_voice
                    if selected_engine == "Sarvam.ai":
                        try:
                            speaker_param = (selected_voice or "").strip().lower()
                        except Exception:
                            speaker_param = selected_voice

                    audio_bytes, error_msg = generate_speech_any(last_response, engine=engine_key, speaker_or_voice=speaker_param, lang=selected_lang)
                    
                    # Show status
                    st.info(error_msg)
                    
                    if audio_bytes:
                        # Provide a BytesIO to st.audio so Streamlit can play it reliably
                        try:
                            audio_buf = io.BytesIO(audio_bytes)
                            audio_buf.seek(0)
                            key_name = f"sanniva_audio_{int(time.time()*1000)}"
                            st.audio(audio_buf, format="audio/mp3", start_time=0, key=key_name)
                        except Exception:
                            # Fallback to passing raw bytes
                            try:
                                st.audio(audio_bytes, format="audio/mp3")
                            except Exception:
                                pass

                        # Also attempt autoplay via an HTML audio tag (browsers may still block autoplay).
                        try:
                            b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                            html_player = f'<audio src="data:audio/mp3;base64,{b64_audio}" autoplay controls></audio>'
                            st.markdown(html_player, unsafe_allow_html=True)
                        except Exception:
                            pass

                        # Offer download button so user can play locally if browser blocks autoplay
                        try:
                            st.download_button("⬇️ Download audio", data=audio_bytes, file_name="sanniva_response.mp3", mime="audio/mpeg")
                        except Exception:
                            pass

                        # Optional: open local player on the server (only useful when running locally)
                        if st.checkbox("Open local player (server)"):
                            try:
                                play_audio_bytes(audio_bytes)
                            except Exception:
                                st.warning("Failed to open local player.")

if __name__ == "__main__":
    main()
