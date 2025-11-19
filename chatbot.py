from google import genai
from google.genai import types
import streamlit as st # type: ignore
import os
import time
from groq import Groq
# import openai
client2 = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
text = client2.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You generate cool and concise Phrases to engage users to my Chatbot application.",
        },
        {
            "role": "user",
            "content": "Generate a catchy phrase to encourage users to interact with a chatbot that helps with anything and roasts them humorously.",
        }
    ],
    model="openai/gpt-oss-20b",
)


client = genai.Client()
text = "Enter what to calculate: "

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def stream_data_to_chat(text: str, delay: float = 0.02):
    placeholder = st.empty()
    current = ""
    for word in text.split(" "):
        current += word + " "
        placeholder.write(current)
        time.sleep(delay)

def get_ai_response(user_input: str, system_prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=[f"User: {user_input}."],
    )
    return response.text if response.text else ""

def display_and_store_response(response_text: str):
    with st.chat_message("assistant"):
        try:
            stream_data_to_chat(response_text)
        except Exception:
            st.write(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

def main():
    st.title("Chat With Me!")
    
    initialize_session_state()
    display_chat_history()
    
    # Get user input
    calc = st.chat_input(text)
    if calc:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": calc})
        with st.chat_message("user"):
            st.write(calc)
        
        # Get system prompt
        with open("System_prompt.txt") as f:
            system_prompt = f.read()
        
        # Get and display AI response
        response_text = get_ai_response(calc, system_prompt)
        display_and_store_response(response_text)

if __name__ == "__main__":
    main()