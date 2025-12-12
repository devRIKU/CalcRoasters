import streamlit as st

 # You will need to install this!

# --- Function to get the User-Agent string ---


# --- Main App Code ---
st.title("🕵️ Website Visitor OS Detector")

# Get the raw User-Agent string
raw_ua = get_user_agent_string()

# Parse the OS
user_os = get_os_from_user_agent(raw_ua)

st.header("Results")
st.code(f"Raw User-Agent: {raw_ua}", language="text")
st.success(f"The User's OS is likely: **{user_os}**")

st.markdown("""
***
### ⚠️ A quick note (the serious part):
To make this code work, you need a special library to read the raw User-Agent string and figure out what the OS is. 

Run this command in your terminal **before** running your Streamlit app:
`pip install user-agents`
""")