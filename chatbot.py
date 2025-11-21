from google import genai
from google.genai import types
import streamlit as st
import os
import time
from groq import Groq

# Initialize Clients
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
client2 = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cache this function so it doesn't slow down the chat on every message
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
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content or "Enter what to calculate: "
    except Exception:
        return "Enter what to calculate: "

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="sanniva_face.jpg"):
            st.markdown(message["content"])

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
        with open("System_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful and humorous assistant."

def get_ai_response_with_brain(prompt: str, system_prompt: str, brain_type: str, chat_history: list) -> str:
    """
    Get AI response based on selected brain type with conversation context.
    
    Args:
        prompt: User input prompt
        system_prompt: System instruction for the AI
        brain_type: Either "Fast" or "Thinker"
        chat_history: List of previous messages for context
    
    Returns:
        AI generated response text
    """
    try:
        if brain_type == "Fast":
            # Build messages with chat history for Groq
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history (limit to last 10 messages to avoid token limits)
            for msg in chat_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = client2.chat.completions.create(
                messages=messages, #type: ignore
                model="llama-3.3-70b-versatile"
            )
            return response.choices[0].message.content or "No response generated."
            
        elif brain_type == "Thinker":
            # Build conversation context for Gemini
            conversation_context = ""
            
            # Add recent chat history (limit to last 10 messages)
            for msg in chat_history[-10:]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role_label}: {msg['content']}\n\n"
            
            # Add current prompt
            full_prompt = f"{conversation_context}User: {prompt}"
            
            # Use Gemini with thinking mode for thoughtful responses
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=8192  # Increased budget for deeper thinking
                    )
                ),
                contents=[full_prompt],
            )
            
            # Handle cases where safety filters block the response
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
    with st.chat_message("assistant"):
        stream_data_to_chat(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

def build_system_prompt(base_prompt: str, personality: str, brain_type: str) -> str:
    """
    Build the complete system prompt with personality and brain modifiers.
    
    Args:
        base_prompt: Base system prompt from file
        personality: Selected personality type ("Roaster" or "Smart")
        brain_type: Selected brain type ("Fast" or "Thinker")
    
    Returns:
        Complete system prompt with modifiers
    """
    prompt = base_prompt
    
    # Add personality modifier
    if personality == "Roaster":
        prompt += " Respond with witty and humorous roasts."
    elif personality == "Smart":
        prompt += " Respond intelligently and thoughtfully."
    
    # Add brain type modifier
    if brain_type == "Thinker":
        prompt += " Take your time to provide thoughtful and detailed responses."
    
    return prompt

def main():
    st.title("Chat With Me!")
    st.sidebar.info("Talk to me, Sanniva's Digital Twin! I can help with anything and roast you humorously.")
    with st.chat_message("assistant", avatar="sanniva_face.jpg"):
        st.write("I'm optimizing my code. What about you?")
    # Personality Selector
    st.sidebar.markdown("**:material/comedy_mask: Personality Selector**")
    personality = st.sidebar.selectbox(
        "",
        ("Roaster", "Smart"),
        key="personality_selector",
        label_visibility="hidden"
    )
    if personality == "Roaster":
        st.sidebar.caption("😂 Roaster personality: Witty and humorous responses")
    elif personality == "Smart":
        st.sidebar.caption("🧠 Smart personality: Intelligent and thoughtful responses")
    
    
    # Brain Selector
    st.sidebar.markdown("**:material/psychology: Brain Selector**")
    brain_type = st.sidebar.selectbox(
        "",
        ("Fast", "Thinker"),
        key="brain_selector",
        label_visibility="hidden"
    )
    
    # Add info about brain types
    if brain_type == "Fast":
        st.sidebar.caption("⚡ Fast mode: Quick responses using Groq")
    else:
        st.sidebar.caption("🧠 Thinker mode: Thoughtful responses using Gemini with extended thinking")
    
    initialize_session_state()
    display_chat_history()
    
    # Get catchy phrase (cached)
    catchy_text = get_catchy_phrase()
    
    # Get user input
    if prompt := st.chat_input(catchy_text):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Load and build system prompt
        base_system_prompt = load_system_prompt()
        system_prompt = build_system_prompt(base_system_prompt, personality, brain_type)
        
        # Get and display AI response with conversation context
        with st.spinner("Thinking..." if brain_type == "Thinker" else "Generating response..."):
            response_text = get_ai_response_with_brain(
                prompt, 
                system_prompt, 
                brain_type, 
                st.session_state.messages
            )
        
        display_and_store_response(response_text)

if __name__ == "__main__":
    main()
