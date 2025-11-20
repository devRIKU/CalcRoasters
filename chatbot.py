from google import genai
from google.genai import types
import streamlit as st
import os
import time
from groq import Groq

# Initialize Clients
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY")) # Ensure API key is passed
client2 = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cache this function so it doesn't slow down the chat on every message
@st.cache_data(ttl=3600) 
def get_catchy_phrase():
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
            model="llama3-8b-8192", # Changed to a reliable model on Groq
        )
        return response.choices[0].message.content
    except Exception:
        return "Enter what to calculate: "

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Use markdown for better formatting

def stream_data_to_chat(text: str, delay: float = 0.02):
    """Streams data with a typing effect."""
    placeholder = st.empty()
    full_response = ""
    
    # Use .split() without arguments to handle newlines/tabs better
    # Or strictly just iterate chunks if you want to preserve exact spacing
    tokens = text.split(" ") 
    
    for i, token in enumerate(tokens):
        full_response += token + " "
        # Add a blinking cursor effect
        placeholder.markdown(full_response + "▌")
        time.sleep(delay)
    
    # Final write without cursor
    placeholder.markdown(full_response)

def get_ai_response(user_input: str, system_prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", # Ensure you use a valid model name
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            ),
            contents=[user_input], # Simplified content passing
        )
        # Handle cases where safety filters block the response
        if response.text:
            return response.text
        else:
            return "I'm speechless. (Safety filters might have blocked my response)."
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_and_store_response(response_text: str):
    with st.chat_message("assistant"):
        stream_data_to_chat(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

def main():
    st.title("Chat With Me!")
    st.sidebar.info("Talk to me, Sanniva's Digital Twin! I can help with anything and roast you humorously.")
    st.sidebar.markdown("**:material/comedy_mask: Personality Selector**")
    lol = st.sidebar.selectbox(
        "",
    ("Roaster","Smart"), key="personality_selector",label_visibility="hidden")
    
    initialize_session_state()
    display_chat_history()
    
    # Get catchy phrase (Now Cached!)
    catchy_text = get_catchy_phrase()
    
    def personality_prompt_modifier(base_prompt: str):
        if lol == "Roaster":
            return f"Respond with humorous roasts. {base_prompt}"
        elif lol == "Smart":
            return f"Respond intelligently and thoughtfully. {base_prompt}"
        return base_prompt
    
    # Get user input
    if prompt := st.chat_input(catchy_text):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get system prompt first
        try:
            with open("System_prompt.txt", "r") as f:
                base_system_prompt = f.read()
        except FileNotFoundError:
            base_system_prompt = "You are a helpful and humorous assistant."
        
        # Apply personality modifier
        system_prompt = personality_prompt_modifier(base_system_prompt)
        
        # Get and display AI response
        with st.spinner("Thinking..."):
            response_text = get_ai_response(prompt, system_prompt)
            
        display_and_store_response(response_text)

if __name__ == "__main__":
    main()
