from google import genai
from google.genai import types
import streamlit as st
import os
import time
from groq import Groq

# --- CONFIGURATION ---
# Make sure these are set in your environment or Streamlit Secrets!
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize Clients with error handling
client = None
client2 = None
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    client2 = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    # Avoid raising at import time; show message once app runs
    st.error(f"Error initializing API clients: {e}. Check your API Keys!")

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
            model="llama-3.3-70b-versatile",
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

def display_chat_history():
    """Display all messages in the chat history."""
    if "messages" not in st.session_state:
        return
    for message in st.session_state.messages:
        avatar = get_avatar() if message.get("role") == "assistant" else None
        with st.chat_message(message.get("role", "user"), avatar=avatar):
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
        if brain_type == "Fast":
            # --- GROQ LOGIC ---
            if client2 is None:
                return "Groq client not initialized."
            messages = [{"role": "system", "content": system_prompt}]
            for msg in chat_history[-10:]:
                messages.append({"role": msg.get("role"), "content": msg.get("content")})
            messages.append({"role": "user", "content": prompt})
            response = client2.chat.completions.create(
                messages=messages,  # type: ignore
                model="llama-3.3-70b-versatile"
            )
            try:
                return response.choices[0].message.content or "No response generated."
            except Exception:
                return str(response) or "No response generated."

        elif brain_type == "Thinker":
            # --- GEMINI LOGIC ---
            if client is None:
                return "Gemini client not initialized."
            conversation_context = ""
            for msg in chat_history[-10:]:
                role_label = "User" if msg.get("role") == "user" else "Assistant"
                conversation_context += f"{role_label}: {msg.get('content')}\n\n"
            full_prompt = f"{conversation_context}User: {prompt}"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=8192
                    ),
                    temperature=temperature,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_ONLY_HIGH"
                        )
                    ]
                ),
                contents=[full_prompt],
            )
            if getattr(response, "text", None):
                return response.text
            try:
                # fallback if response structure differs
                return str(response)
            except Exception:
                return "I'm speechless. (Safety filters might have blocked my response)."
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

    if personality == "Roaster":
        st.sidebar.caption("😂 **Roaster:** Witty & Savage")
    elif personality == "Smart":
        st.sidebar.caption("🧠 **Smart:** Intelligent & Polite")
    elif personality == "Debater":
        st.sidebar.caption("🎓 **Debater:** Debates Against Anything")

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

    display_chat_history()

    # (No auto-personality) — personality comes from the sidebar selectbox

    catchy_text = get_catchy_phrase()

    if prompt := st.chat_input(catchy_text):
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

if __name__ == "__main__":
    main()
