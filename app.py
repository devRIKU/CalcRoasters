from google import genai
from google.genai import types
import streamlit as st

client = genai.Client()

st.title("Roasting Calculator Assistant")

# Get user input via Streamlit
calc = st.text_input("Enter what to calculate: ", key="calc", value="")
st.chat_message("user").write("Now do it for me:")
work = st.text_input("answer: ", key="work", value="")

# Only call API if both inputs are provided
if calc and work:
    system_prompt = f"You are never an helpful assistant that helps people to do calculations. When given a calculation, you roast the user twice and judge their {calc} critically and tell user to do it your self. For example, if the user provides the calculation '{calc}', you should roast that person then after roasting quite a bit give the answer."
    contents = f"Calculation {calc}, work: {work}."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=[contents],
    )
    st.write("Response:", response.text)