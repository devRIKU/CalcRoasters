# Sanniva — a two-brained Streamlit chatbot

A personality-driven Streamlit chatbot built around a **"digital twin"** persona
(Sanniva, a Class 8 student in West Bengal). Two interchangeable LLM backends,
per-user lore memory, tool/function calling, and free text-to-speech with no
keys required.

> Repo / module: `calcroasters` &nbsp;|&nbsp; Entry point: `chatbot.py`

---

## Highlights

- **Two brains, on demand**
  - **Fast (Groq)** — streams tokens for snappy chat; handles tool calls
    mid-stream.
  - **Thinker (Gemini)** — non-streaming, deeper reasoning for harder
    questions.
- **Automatic model fallback** — each brain has an ordered list of model IDs.
  If the first one 404s / rate-limits / times out, the next is tried.
- **7 personality modes** — Roaster, Smart, Debater, Strategic, Tech Nerd,
  Chill Squad, Exhausted Student. Each remaps the UI theme in real time.
- **Tool / function calling** — the model can:
  - `request_user_name` — open a Streamlit dialog asking for the user's name
  - `remember_lore(user_name, fact)` — save a fact to public + private store
  - `recall_lore(user_name)` — fetch facts before answering
- **Inline tool-call recovery** — some Groq and Gemini flash builds emit tool
  calls as `<function=...>{...}</function>` text instead of structured
  `tool_calls`. Sanniva parses these out of `content` and executes them
  anyway, so the user never sees the broken pseudo-XML.
- **Per-user lore memory** — public facts in `lore.json`; private facts in
  **Firebase Firestore** (if configured) with **SQLite** (`private_lore.db`)
  as a zero-config local fallback.
- **Free TTS by default** — Edge TTS (Microsoft) and gTTS (Google Translate)
  work with no API key. Paid engines (Sarvam.ai, Fish Audio, SiliconFlow)
  light up automatically when their key is set.
- **Auto-play, compact player, or `st.audio`** — three rendering modes for
  the audio response, plus an optional download button.
- **Live temporal context** — date, grade level, and West Bengal academic
  phase are auto-injected into the system prompt and re-evaluated on each
  request, so the persona ages with the calendar.

---

## Quickstart

### 1. Prerequisites

- Python **3.10+** (3.11 / 3.12 / 3.13 all work)
- One LLM key (both recommended): a **Groq** key and/or a **Google AI Studio**
  (Gemini) key. The free tiers are enough to play with.

### 2. Clone and install

```bash
git clone https://github.com/<your-fork>/calcroasters.git
cd calcroasters
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` in the repo root. **At minimum one of `GROQ_API_KEY` or
`GOOGLE_API_KEY` is required** — everything else is optional.

```ini
# --- LLMs (need at least one) ---
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...

# --- Optional paid TTS engines (free Edge/gTTS always work without these) ---
# SARVAM_API_KEY=...
# FISH_AUDIO_API_KEY=...
# SILICON_FLOW_API_KEY=...

# --- Optional Firebase (private per-user lore). Falls back to local SQLite if absent. ---
# Easiest: paste the entire service-account JSON on one line:
# FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
# Or set the three fields individually:
# FIREBASE_PROJECT_ID=...
# FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# FIREBASE_CLIENT_EMAIL=...@....iam.gserviceaccount.com
```

See `README_ENV.md` for the long-form setup notes.

### 4. Run

```bash
streamlit run chatbot.py
```

Browse to <http://localhost:8501>.

---

## Using the app

The sidebar groups settings into five sections:

| Section | What it controls |
|---|---|
| **Identity** | Your display name + expandable view of what Sanniva remembers about you. |
| **Personality** | Pick one of seven modes; the UI re-themes itself instantly. |
| **Brain Power** | `Fast` (Groq) or `Thinker` (Gemini). |
| **Model settings** | Clear chat, creativity slider, per-model timeout, and the comma-separated fallback chain for each brain. |
| **TTS Engine** | Engine selector (paid engines only appear if their key is set), voice / language, plus auto-play, HTML-autoplay, and compact-icon toggles. |

Talk to it in the chat box at the bottom. The model may decide to call a tool
mid-conversation (e.g. asking for your name, storing a fact); those run
transparently and the visible answer is the natural-language wrap-up.

### Default model fallback chains

**Groq (Fast)**

1. `llama-3.3-70b-versatile` — 300K TPM free tier, tool-calling
2. `openai/gpt-oss-120b` — 250K TPM, tool-calling
3. `llama-3.1-8b-instant` — fastest, lowest TPM cap

**Gemini (Thinker)**

1. `gemini-3.5-flash` — newest GA flash, strongest reasoning
2. `gemini-3-flash-preview` — preview build of the 3.x flash line
3. `gemini-3.1-flash-lite` — cheapest, fastest fallback

Both lists are editable in the sidebar at runtime — useful when a model ID
404s or you want to try something new.

---

## File map

| File | Purpose |
|---|---|
| `chatbot.py` | Main Streamlit app: UI, LLM orchestration, tool dispatch, TTS routing, sidebar. |
| `tts_free.py` | Free TTS providers (Edge, gTTS). Voice catalog `EDGE_VOICES`. |
| `tools.py` | Tool schemas (OpenAI / Gemini shapes) + `dispatch()` / `dispatch_json()` runners. |
| `lore_store.py` | Per-user memory: `lore.json` (public) + Firestore or SQLite (private). |
| `styles.py` | Writes `.streamlit/config.toml` colour theme per personality. |
| `System_prompt.md` | The long persona spec (~17 kB) injected as the system message. |
| `test_tts.py` | Standalone smoke test for the free TTS engines (writes `test_edge.mp3` / `test_gtts.mp3`). |
| `sanniva_face.jpg` | Avatar shown in chat. |
| `requirements.txt` | Pinned dependencies. |
| `README_ENV.md` | Extended env-var / deployment notes. |

---

## How tool calling works

`tools.py` exposes two parallel schema builders:

- `OPENAI_TOOLS` — JSON schemas in the OpenAI / Groq shape.
- `build_gemini_tools()` — equivalent schemas in Google's `genai.types.Tool` shape.

The Groq path uses a bounded tool-calling loop (`MAX_TOOL_HOPS = 4`):

1. First hop is **non-streaming** with `tool_choice="auto"` so we can inspect
   `tool_calls`.
2. If structured tool calls come back → run them, append the results, loop.
3. If the model emits pseudo-XML (`<function=name:foo {...}</function>`) inside
   `content` instead of using `tool_calls`, the recovery parser at
   `chatbot.py:_extract_inline_tool_calls` lifts them out and treats them as
   real calls.
4. Once no tools are pending, the final answer is **streamed** with tools
   disabled so the model is forced to produce prose.

The Gemini path uses the same idea but with `function_call` parts on
`response.candidates[*].content.parts`.

---

## Troubleshooting

### "rate_limit" / "Limit 6000" from Groq
The free tier on `llama-3.1-8b-instant` is 6 000 TPM, and Sanniva's system
prompt is ~17 kB. Either wait 60 s, move that model lower in the chain, or
shorten the prompt.

### Gemini returns 404 on `gemini-3-flash-preview`
The preview tier may not be enabled on your Google project. The fallback
loop will skip it and try `gemini-3.1-flash-lite` automatically — no action
needed.

### TTS produces a blank audio file
Make sure `edge-tts` is installed (`pip install edge-tts`) and that you have
outbound HTTPS to `speech.platform.bing.com`. Run the smoke test:

```bash
python test_tts.py
```

You should get a `test_edge.mp3` of ~10 KB and a `test_gtts.mp3` of ~18 KB.
If the smoke test works but the in-app TTS doesn't, check the sidebar TTS
section for a red error banner — the app now surfaces TTS errors explicitly
rather than failing silently.

### Firebase not connecting
Either the JSON is malformed or one of `FIREBASE_PROJECT_ID` /
`FIREBASE_PRIVATE_KEY` / `FIREBASE_CLIENT_EMAIL` is missing. The app keeps
running on the local SQLite fallback — check the sidebar for the init
warning if you're expecting cloud sync.

---

## Development notes

- `chatbot.py` is intentionally a single file. The sidebar, theme, brain
  loops, and TTS dispatch are all in one place to keep state management
  simple under Streamlit's rerun-everything model.
- `.streamlit/config.toml` is **rewritten on every personality switch** by
  `styles.py`. If you're versioning theme changes, be aware Streamlit will
  overwrite manual edits the next time the user changes personality.
- Anything written to `lore.json` is **public** (visible to anyone using the
  app). Use the `add_fact(..., private=True)` path in `lore_store.py` for
  per-user data.
- The system prompt is loaded fresh from `System_prompt.md` on each request,
  so editing the persona doesn't require a server restart.

---

## License

No license file is included yet. Until one is added, treat the contents as
**all rights reserved** — fine for personal experimentation, but please ask
before redistributing.
