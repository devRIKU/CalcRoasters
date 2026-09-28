Quick env setup for CalcRoasters

1) Copy the example file to create your local `.env` (recommended):

PowerShell:

```powershell
copy .env.example .env
```

2) Open `.env` and fill in your API keys:
- `GOOGLE_API_KEY` (Gemini)
- `GROQ_API_KEY`
- `COHERE_API_KEY`
- `OPENROUTER_API_KEY`
- `SARVAM_API_KEY` (optional, paid TTS)
- `FISH_AUDIO_API_KEY` (optional, paid TTS)
- `SILICON_FLOW_API_KEY` (optional, paid TTS)

3) Activate your virtual environment (Windows PowerShell):

```powershell
& .venv\Scripts\Activate.ps1
```

4) Run the app:

```powershell
streamlit run chatbot.py
```

Notes:
- Do NOT commit your `.env` to Git. Add it to `.gitignore` if needed.
- If you prefer, set environment variables directly in your OS instead of using a `.env` file.
