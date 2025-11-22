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
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    client2 = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Error initializing API clients: {e}. Check your API Keys!")

# Helper for Avatar
def get_avatar():
    """Returns image path if exists, else emoji"""
    if os.path.exists("sanniva_face.jpg"):
        return "sanniva_face.jpg"
    return "🤖" # Fallback emoji

@st.cache_data(ttl=3600)
def get_catchy_phrase() -> str:
    """Generate a catchy phrase using Groq."""
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
            model="llama-3.3-70b-versatile", # Updated to match your main logic
        )
        return response.choices[0].message.content or "Yeah go ahead, ask me anything!"
    except Exception:
        return "Yeah go ahead, ask me anything!"

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        # Check if it's assistant to apply avatar
        avatar = get_avatar() if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def stream_data_to_chat(text: str, delay: float = 0.02):
    """Streams data with a typing effect."""
    placeholder = st.empty()
    full_response = ""
    
    # Split by words to keep it smoother
    tokens = text.split(" ") 
    
    for token in tokens:
        full_response += token + " "
        placeholder.markdown(full_response + "▌")
        time.sleep(delay)
    
    placeholder.markdown(full_response)

def load_system_prompt() -> str:
    """Load the base system prompt from file."""
    try:
        with open("System_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful and humorous assistant named Sanniva."

def get_ai_response_with_brain(prompt: str, system_prompt: str, brain_type: str, chat_history: list) -> str:
    """Get AI response based on selected brain type."""
    try:
        if brain_type == "Fast":
            # --- GROQ LOGIC ---
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history (last 10)
            for msg in chat_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            messages.append({"role": "user", "content": prompt})
            
            response = client2.chat.completions.create(
                messages=messages, #type: ignore
                model="llama-3.3-70b-versatile"
            )
            return response.choices[0].message.content or "No response generated."
            
        elif brain_type == "Thinker":
            # --- GEMINI LOGIC ---
            conversation_context = ""
            
            # Format history as a string context for Gemini
            for msg in chat_history[-10:]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role_label}: {msg['content']}\n\n"
            
            full_prompt = f"{conversation_context}User: {prompt}"
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite", 
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=8192
                    ),
                    # Loosen safety for "Roaster" mode to work
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
            
            if response.text:
                return response.text
            else:
                return "I'm speechless. (Safety filters might have blocked my response)."
        else:
            return "Invalid brain type selected."
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_and_store_response(response_text: str):
    """Display AI response with streaming effect and store in session."""
    with st.chat_message("assistant", avatar=get_avatar()):
        stream_data_to_chat(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

def build_system_prompt(base_prompt: str, personality: str, brain_type: str) -> str:
    prompt = base_prompt
    
    if personality == "Roaster":
        prompt += " You are in ROAST MODE. Be witty, savage, and roast the user humorously based on their input."
    elif personality == "Smart":
        prompt += " Respond intelligently, academically, and thoughtfully."
    
    if brain_type == "Thinker":
        prompt += " Use deep thinking to analyze the request before answering."
    
    return prompt

def main():
    st.set_page_config(page_title="Sanniva AI", page_icon="🤖") # Added page config
    st.title("Chat With Sanniva!")
    
    st.sidebar.info("I am Sanniva's Digital Twin! I can help with anything and roast you humorously.")
    
    # Only show this once using session state logic if you want, but static is fine
    # with st.sidebar:
    #     st.image(get_avatar(), width=100) if get_avatar() != "🤖" else st.write("🤖")
    
    # Personality Selector
    st.sidebar.markdown("**:material/comedy_mask: Personality**")
    personality = st.sidebar.selectbox(
        "Select Personality",
        ("Roaster", "Smart"),
        key="personality_selector",
        label_visibility="collapsed"
    )
    
    if personality == "Roaster":
        st.sidebar.caption("😂 **Roaster:** Witty & Savage")
    elif personality == "Smart":
        st.sidebar.caption("🧠 **Smart:** Intelligent & Polite")
    
    # Brain Selector
    st.sidebar.markdown("**:material/psychology: Brain Power**")
    brain_type = st.sidebar.selectbox(
        "Select Brain",
        ("Fast", "Thinker"),
        key="brain_selector",
        label_visibility="collapsed"
    )
    
    if brain_type == "Fast":
        st.sidebar.caption("⚡ **Fast:** Instant answers (Groq)")
    else:
        st.sidebar.caption("🕵️ **Thinker:** Deep reasoning (Gemini 2.0 Thinking)")
    
    initialize_session_state()
    display_chat_history()
    
    catchy_text = get_catchy_phrase()
    
    if prompt := st.chat_input(catchy_text):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # System Prompt & Response
        base_system_prompt = load_system_prompt()
        system_prompt = build_system_prompt(base_system_prompt, personality, brain_type)
        
        with st.spinner("Thinking..." if brain_type == "Thinker" else "Generating..."):
            response_text = get_ai_response_with_brain(
                prompt, 
                system_prompt, 
                brain_type, 
                st.session_state.messages
            )
        
        display_and_store_response(response_text)

if __name__ == "__main__":
    main()