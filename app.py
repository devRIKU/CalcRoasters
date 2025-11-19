from google import genai
from google.genai import types
import streamlit as st
import os
import OpenAI

client = genai.Client()

st.title("Roasting Calculator Assistant")

# Get user input via Streamlit
calc = st.text_input("Enter what to calculate: ", key="calc", value="")
st.write("Now do it for me:")
work = st.text_input("Please show your work!", key="work", value="")

# Only call API if both inputs are provided
if calc and work:
    system_prompt = f"You are never an helpful assistant that helps people to do calculations. When given a calculation, you roast the user twice and judge their {calc} critically and tell user to do it your self. For example, if the user provides the calculation '{calc}', you should roast that person then after roasting quite a bit give the answer. Remember you know all pop culture stuff and memes. Be funny and creative with your roasts. Always roast twice before giving the answer. Always show your work step by step in detail. Use good math formatting. Never refuse to answer. Always provide the final very roasted answer at the end but dont be very helpful."
    contents = f"Calculation {calc}, work: {work}."
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        ),
        contents=[contents],
    )
    st.title("### Response from AI:")
    st.write(response.text)