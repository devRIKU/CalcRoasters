from google import genai
from google.genai import types
import streamlit as st
import os
import time
# import openai

client = genai.Client()
# NOTE: removed automatic file upload of system instructions — keep System_prompt.txt in repo root
st.title("Chat With Me!")
file = client.files.upload("System_prompt.txt")
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input via Streamlit
calc = st.chat_input("Enter what to calculate: ")
if calc:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": calc})
    with st.chat_message("user"):
        st.write(calc)

    # Only call API if input is provided
    with open("System_prompt.txt") as f:
        system_prompt = f.read()
    contents = f"User: {calc}."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=[contents],
    )
    res = response.text if response.text else ""

    # stream the assistant response into a new chat message using a placeholder
    def stream_data_to_chat(text: str, delay: float = 0.02):
        placeholder = st.empty()
        current = ""
        for word in text.split(" "):
            current += word + " "
            # Replace placeholder contents on each step so the message looks streamed
            placeholder.write(current)
            time.sleep(delay)
    
    # Add AI message to history
    # Display streaming assistant message, then persist to session history
    with st.chat_message("assistant"):
        try:
            stream_data_to_chat(res)
        except Exception:
            # fallback to a single write if the stream fails
            st.write(res)

    # Persist final assistant content into history after streaming completes
    st.session_state.messages.append({"role": "assistant", "content": res})