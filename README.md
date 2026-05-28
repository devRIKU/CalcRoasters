# Sanniva — a four-provider Streamlit chatbot

A personality-driven Streamlit chatbot built around a **"digital twin"** persona
(Sanniva, a middle-school student in West Bengal). Four interchangeable LLM
backends with automatic cross-provider failover, per-user lore memory (public
vs private), parallel tool-calling, prompt-caching cost telemetry, and free
text-to-speech with no keys required.

> Repo / module: `calcroasters` &nbsp;|&nbsp; Entry point: `chatbot.py`

---

## Highlights

- **Four LLM providers with auto-failover.** The app talks to **Groq**,
  **Gemini**, **Cohere (v2)**, and **OpenRouter** (which itself gives you
  access to 300+ models behind one key). If your preferred provider's models
  all fail, the dispatcher silently walks to the next provider in the
  fallback chain — the user sees no dead chat.
- **Brain Power is a preference, not a lock.**
  - **⚡ Fast:** prefers `Groq → OpenRouter → Cohere → Gemini`. Best for
    streaming, low-latency replies, snappy first-token.
  - **🕵️ Thinker:** prefers `Gemini → Cohere → OpenRouter → Groq`. Best for
    deeper reasoning, harder questions, longer context.
- **Provider attribution on every turn.** Each assistant message gets a
  small footer caption: `via 🪶 Cohere · 1.2s`. You always know who actually
  served the response, even after a failover.
- **Cache savings + spend telemetry in the sidebar.** The `💰 Cache savings`
  expander shows per-provider hit rates, effective billed tokens, and (for
  OpenRouter) running dollar cost — so you can verify caching is actually
  hitting and catch surprise spend immediately.
- **7 personality modes** — Roaster, Smart, Debater, Strategic, Tech Nerd,
  Chill Squad, Exhausted Student. Each remaps the UI theme in real time.
- **Per-user lore memory** — public facts in `lore.json`; private facts in
  **Firebase Firestore** (if configured) with **SQLite** (`private_lore.db`)
  as a zero-config local fallback. The model is taught when to use each via
  the `private: true/false` parameter on `remember_lore`.
- **Parallel tool calling.** Multiple tools in one hop run on a shared
  `ThreadPoolExecutor`; the model's leading prose streams to the UI
  *immediately* while tools run in the background.
- **Inline tool-call recovery** — some Groq / Gemini / OpenRouter flash
  models emit tool calls as `<function=...>{...}</function>` text instead
  of structured `tool_calls`. The app parses these out and executes them
  anyway, so the user never sees the broken pseudo-XML.
- **Free TTS by default** — Edge TTS (Microsoft) and gTTS (Google
  Translate) work with no API key. Paid engines (Sarvam.ai, Fish Audio,
  SiliconFlow) light up automatically when their key is set. TTS strips
  markdown / stage directions so the voice doesn't read `*sighs*` as
  "asterisk sighs asterisk".
- **Live temporal context** — date, grade level, and West Bengal academic
  phase are auto-injected into the system prompt and re-evaluated on each
  request, so the persona ages with the calendar.

---

## Quickstart

### 1. Prerequisites

- Python **3.10+** (3.11 / 3.12 / 3.13 all work)
- **At least one** LLM API key. The four supported providers are listed
  below — you can wire up any subset and the rest just stay disabled.

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

Create a `.env` in the repo root. **At least one LLM provider key is
required** — every other variable is optional.

```ini
# --- LLM providers (at least one required) ---
# Groq: fast streaming, OpenAI-shape API, generous free tier.
GROQ_API_KEY=gsk_...

# Google AI Studio (Gemini): best reasoning, implicit caching at 75% off.
GOOGLE_API_KEY=AIza...

# Cohere v2: Command-A / Command-R Plus families, strong on tool use.
COHERE_API_KEY=...

# OpenRouter: one key, 300+ upstream models (Claude, GPT, Llama, Mistral,
# Qwen, DeepSeek, Kimi, …). Free-tier models are used by default — paid
# models can be added explicitly via the sidebar's "Add custom model".
OPENROUTER_API_KEY=sk-or-v1-...
# Optional ranking metadata sent on every OpenRouter request:
OPENROUTER_REFERER=https://your-deploy-url
OPENROUTER_TITLE=Your App Name

# --- Optional paid TTS engines (free Edge/gTTS always work without these) ---
# SARVAM_API_KEY=...
# FISH_AUDIO_API_KEY=...
# SILICON_FLOW_API_KEY=...

# --- Optional Firebase (private per-user lore). Falls back to local SQLite. ---
# Easiest: paste the entire service-account JSON on one line:
# FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
# Or set the three fields individually:
# FIREBASE_PROJECT_ID=...
# FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# FIREBASE_CLIENT_EMAIL=...@....iam.gserviceaccount.com
```

See `README_ENV.md` for long-form deployment notes.

### 4. Run

```bash
streamlit run chatbot.py
```

Browse to <http://localhost:8501>.

---

## Provider matrix

| Provider | SDK | Streaming | Tools | Cache discount | Notes |
|---|---|---|---|---|---|
| **Groq** | `groq` | ✅ | ✅ | 50% (automatic) | Fastest streaming. Best for the Fast brain. |
| **Gemini** | `google-genai` | non-stream today | ✅ | 75% (2.5+/3.x implicit) | Strongest reasoning. Best for the Thinker brain. |
| **Cohere v2** | `cohere` | ✅ | ✅ | N/A (no documented discount) | Command-A / Command-R Plus. Excellent at structured tool use. |
| **OpenRouter** | `openai` (pointed at `openrouter.ai/api/v1`) | ✅ | ✅ | 50% (upstream caching) | 300+ models behind one key. Free-tier defaults. **Spend surfaced in sidebar.** |

The sidebar gives each provider its own model multiselect (Cohere → Groq →
Gemini → OpenRouter, top to bottom), plus an `➕ Add custom model` expander
for IDs not in the live catalogue.

---

## Using the app

The sidebar groups settings into seven sections:

| Section | What it controls |
|---|---|
| **Identity** | Your display name + expandable view of what Sanniva remembers about you (split into 🌐 Public lore and 🔒 Private lore). |
| **Personality** | Pick one of seven modes; the UI re-themes itself instantly. |
| **Brain Power** | `Fast` or `Thinker` — quality dial, not a provider lock. Below the selector you'll see a live row of provider status pills. |
| **Model settings** | Clear chat, creativity slider, per-model timeout. |
| **Model Fallback Chains** | One multiselect per provider (Cohere → Groq → Gemini → OpenRouter). Each provider tries its models in order; failure walks to the next provider in the brain's preferred chain. |
| **💰 Cache savings** | Per-provider hit rates, effective billed tokens, OpenRouter dollar spend. |
| **TTS Engine** | Engine selector (paid engines only appear if their key is set), voice / language, plus auto-play, HTML-autoplay, and compact-icon toggles. |

Talk to it in the chat box at the bottom. When the model calls a tool, a
short "🔧 Running …" caption appears above the input while the tool
executes in parallel; the model's leading prose ("on it — let me check…")
streams into the chat *immediately* so the UI never sits frozen.

### Default model fallback chains

**🪶 Cohere** (newest first)
1. `command-a-plus-05-2026` — newest Command-A Plus, top capability
2. `command-r-plus-08-2024` — older but reliable, strong tool use

**⚡ Groq**
1. `llama-3.3-70b-versatile` — 300K TPM free tier, tool-calling
2. `openai/gpt-oss-120b` — 250K TPM, tool-calling

> `llama-3.1-8b-instant` is **not** in the default chain — its 6 000 TPM
> cap is too low for the ~10 KB system prompt and was the source of
> misleading rate-limit errors. It's still available via the picker.

**🕵️ Gemini**
1. `gemini-3.5-flash` — newest GA flash, strongest reasoning
2. `gemini-3-flash-preview` — preview build of the 3.x flash line
3. `gemini-3.1-flash-lite` — cheapest, fastest fallback

**🎛️ OpenRouter** (free-tier only by default — add paid via sidebar)
1. `x-ai/grok-4-fast:free`
2. `openai/gpt-oss-120b:free`
3. `deepseek/deepseek-chat-v3.1:free`
4. `meta-llama/llama-3.3-70b-instruct:free`

All four lists are editable in the sidebar at runtime — useful when a
model ID 404s or you want to try something new.

---

## Auto-failover

When a model fails (rate limit, network blip, model decommissioned,
streaming error), the dispatcher does this:

1. **Within a provider:** the next model in that provider's chain is tried.
2. **Across providers:** if every model in a provider fails, the
   dispatcher walks to the next provider in the brain's preferred order.

Example — `Fast` brain with all providers wired up, Groq down:

```
Groq/llama-3.3-70b-versatile  → rate-limited
Groq/openai/gpt-oss-120b      → timeout
   [Groq exhausted, falling over to OpenRouter]
OpenRouter/x-ai/grok-4-fast:free → ✅ served the response
```

The user sees their answer; the sidebar's provider attribution caption
shows `via 🎛️ OpenRouter · 1.4s`. No banner, no error, no "try again".

---

## File map

| File | Purpose |
|---|---|
| `chatbot.py` | Main Streamlit app: UI, LLM orchestration, tool dispatch, TTS routing, sidebar. |
| `tts_free.py` | Free TTS providers (Edge, gTTS). Voice catalog `EDGE_VOICES`. |
| `tools.py` | Tool schemas (OpenAI / Gemini shapes), `dispatch()` / `dispatch_json()` / `dispatch_parallel()` runners. |
| `lore_store.py` | Per-user memory: `lore.json` (public) + Firestore or SQLite (private). |
| `styles.py` | Writes `.streamlit/config.toml` colour theme per personality. |
| `System_prompt.md` | The persona spec injected as the system message. Includes privacy gate, Friend Mode, family, interests, online presence. |
| `test_tts.py` | Standalone smoke test for the free TTS engines. |
| `sanniva_face.jpg` | Avatar shown in chat. |
| `requirements.txt` | Pinned dependencies (Streamlit, groq, google-genai, cohere, openai, etc.). |
| `README_ENV.md` | Extended env-var / deployment notes. |

---

## How tool calling works

`tools.py` exposes:

- `OPENAI_TOOLS` — JSON schemas in the OpenAI / Groq / Cohere v2 / OpenRouter shape (they all accept the same `{"type": "function", "function": {...}}` envelope).
- `build_gemini_tools()` — the equivalent schemas as Google's `genai.types.Tool` objects.
- `dispatch(name, args)` — runs a single tool, returns a dict.
- `dispatch_json(name, args_json)` — string in / string out, for Groq/OpenRouter `role:"tool"` messages.
- `dispatch_parallel(calls)` — runs N tools concurrently on a shared `ThreadPoolExecutor(max_workers=4)`.

The OpenAI-compatible providers (Groq, OpenRouter) use a bounded loop
(`MAX_TOOL_HOPS = 4`):

1. **First hop** is non-streaming with `tool_choice="auto"` — we need the
   structured `tool_calls` array.
2. **If tool calls are present:** the model's leading prose is yielded to
   the UI *immediately* (instant feedback); tools run in parallel; results
   are appended to the message history.
3. **Final hop** is streamed with tools disabled so the model produces
   prose. Tokens are word-aligned by `_word_chunk_stream` for a smooth
   typewriter animation even when chunks arrive jagged.

Cohere v2 follows the same shape via `chat()` for the first hop and
`chat_stream()` for the final synthesis, with its own event-type handling
(`content-delta`, `tool-call-start`, etc.).

Gemini uses `function_call` parts on `response.candidates[*].content.parts`
and is currently non-streaming end-to-end; the UI animates its finished
string with the same word-by-word effect.

If a model emits inline pseudo-XML (`<function=name:foo {...}</function>`)
instead of structured tool calls, `_extract_inline_tool_calls` in
`chatbot.py` recovers them and treats them as real calls.

---

## Cost & cache observability

The sidebar's **💰 Cache savings** expander reads token usage off every
completion and shows per-provider:

- Session input tokens, cached tokens, hit-rate %
- Last turn's tokens + cache breakdown
- Effective billed tokens with the documented discount applied
- **For OpenRouter:** running dollar spend (session + last turn).
  Free-tier models report `$0.00`; paid models report real cost from the
  `usage.cost` field OpenRouter attaches when you set `usage: {include: true}`
  on the request (the app sets this automatically).

| Provider | Discount on cached input | Where you see it |
|---|---|---|
| Groq | 50% | sidebar widget |
| OpenRouter (upstream caching) | typically 50% | sidebar widget + $ cost |
| Gemini (implicit, 2.5+/3.x) | 75% | (telemetry capture pending) |
| Cohere | no documented discount | sidebar shows raw token counts |

---

## Troubleshooting

### "All providers failed" error
This only fires when every provider in the brain's chain exhausted with
no successful response. Check:
1. Are any of the four API keys set? See the sidebar provider pills — a
   ❌ means key missing / init failed.
2. Did all providers' models 404? Open the model-fallback expander for
   each provider and confirm the model IDs look current.
3. Network — `curl https://api.groq.com/openai/v1/models` etc.

### "rate_limit" / "Limit 6000" from Groq
The free tier on `llama-3.1-8b-instant` is 6 000 TPM. The default chain
doesn't include it, but if you added it back via the picker, the system
prompt alone may exceed the cap. Remove it from the chain or wait 60 s.

### OpenRouter cost suddenly > $0.00
You added a paid model via the sidebar's "Add custom OpenRouter model"
expander. Check the `💰 Cache savings` expander's "OpenRouter spend"
line for the running session total.

### Gemini returns 404 on `gemini-3-flash-preview`
The preview tier may not be enabled on your Google project. The fallback
loop will skip it and try `gemini-3.1-flash-lite` automatically — no
action needed.

### Cohere "no model selected"
The picker defaults to `command-a-plus-05-2026, command-r-plus-08-2024`.
If you deselected both, the dispatcher will skip Cohere entirely on
failover. Re-add at least one model in the picker.

### TTS produces a blank audio file
Make sure `edge-tts` is installed (`pip install edge-tts`) and that you
have outbound HTTPS to `speech.platform.bing.com`. Run the smoke test:

```bash
python test_tts.py
```

You should get `test_edge.mp3` (~10 KB) and `test_gtts.mp3` (~18 KB). If
the smoke test works but the in-app TTS doesn't, check the sidebar TTS
section for a red error banner — the app surfaces TTS errors explicitly
rather than failing silently.

### Firebase not connecting
Either the JSON is malformed or one of `FIREBASE_PROJECT_ID` /
`FIREBASE_PRIVATE_KEY` / `FIREBASE_CLIENT_EMAIL` is missing. The app keeps
running on the local SQLite fallback — check the sidebar for the init
warning if you're expecting cloud sync.

---

## Development notes

- **`chatbot.py` is intentionally a single file.** The sidebar, theme,
  brain loops, and TTS dispatch are all in one place to keep state
  management simple under Streamlit's rerun-everything model.
- **Provider order is wired in two places:**
  - `_provider_order_for_brain(brain_type)` — returns the preferred
    failover order for `Fast` vs `Thinker`.
  - `PROVIDER_ORDER` — global default (unused for the user-facing chain
    today, but kept as the canonical ordering reference).
- **The system prompt's persona-mode block is trimmed at load time** so
  only the active mode is sent (saves ~1,300 tokens per turn). The full
  `System_prompt.md` is still the human-editable source.
- **Cache-friendly prompt ordering:** `build_system_prompt` puts static
  content first (base file → persona suffix → tool guidance) and pushes
  per-turn variability (user name, lore block, popup state, temporal
  context) to the end. This keeps Groq's automatic prefix caching
  hitting across turns within a session.
- **`.streamlit/config.toml` is rewritten on every personality switch**
  by `styles.py`. If you're versioning theme changes, be aware Streamlit
  will overwrite manual edits the next time the user changes personality.
- **Anything written to `lore.json` is public** (visible to anyone using
  the app). The `add_fact(..., private=True)` path goes to Firestore /
  SQLite instead. The model decides which bucket per-fact via the
  `private` parameter on `remember_lore`, with sensitivity guidance in
  the tool description.

---

## License

No license file is included yet. Until one is added, treat the contents
as **all rights reserved** — fine for personal experimentation, but
please ask before redistributing.
