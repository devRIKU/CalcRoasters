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
- **Provider attribution on every turn.** Each assistant message records
  the provider that actually served it (`provider`, `elapsed_s`) — even
  after a failover — and the sidebar's status pills / cache widget surface
  it (`last: 🪶 Cohere`). The inline footer caption was dropped for a
  cleaner chat, the provenance is still there.
- **Cache savings + spend telemetry in the sidebar.** The `💰 Cache savings`
  expander shows per-provider hit rates, effective billed tokens, and (for
  OpenRouter) running dollar cost — so you can verify caching is actually
  hitting and catch surprise spend immediately.
- **7 personality modes** — Roaster, Smart, Debater, Strategic, Tech Nerd,
  Chill Squad, Exhausted Student. Each remaps the UI theme in real time.
- **Two gender variants + a live transition button.** The persona ships as a
  **♂️ male** and a **♀️ female** system prompt (`System_prompt.md` /
  `System_prompt_female.md`). The sidebar's `🔁 Transition` button
  gender-swaps the *active* conversation **without clearing anything** — the
  app fires a system event telling the LLM it was swapped mid-chat, so
  memories, lore and the whole transcript carry over. A toggle directly
  below it decides the gender a *new* conversation starts in.
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
| **Gender / Persona Identity** | `🔁 Transition to ♂️/♀️` button (mid-chat gender swap that keeps context) + `♀️ Start new chats as the correct gender` toggle directly below it. |
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

### Gender variants & the transition event

The persona exists in two full system prompts:

| Variant | File | Pronouns |
|---|---|---|
| ♂️ Male | `System_prompt.md` | he/him |
| ♀️ Female (the real Sanniva) | `System_prompt_female.md` | she/her |

Two sidebar controls manage which one is active:

- **`🔁 Transition to …`** — swaps the active variant *mid-conversation*.
  `st.session_state.messages` (the context), lore, tools and every other bit
  of session state are left untouched. What the button actually does:
  1. sets `gender_mode` to the other variant,
  2. appends a `system_note` event bubble to the transcript
     (`🔁 Transition event — … Context preserved …`),
  3. appends a record to `_gender_events`, which
     `build_system_prompt()` renders as a
     `## ⚡ LIVE SYSTEM EVENT — GENDER TRANSITION` block **on every
     subsequent turn** (it sits in the prompt's variable tail, so prompt
     caching is unaffected),
  4. sends one extra in-band `[[SYSTEM EVENT — GENDER TRANSITION]]` message
     on the very next request.

  The LLM is explicitly told: nothing else changed, do not re-introduce
  yourself, do not recap, keep going in the same turn, and only mention the
  swap if the user brings it up.
- **`♀️ Start new chats as the correct gender (female)`** — the toggle
  *below* the transition button. It only affects conversations that haven't
  started yet (or after `🗑️ Clear Chat`); flipping it mid-chat is a no-op and
  the sidebar says so, because switching a running conversation is what the
  Transition button is for.

Both variants are keyed into the same prompt-caching layout: static content
first, the transition block last, so a swap costs one cache miss, not one
per turn.

### Default model fallback chains

*Verified 2026-09-28 directly against each provider's own model docs /
catalogue (see the links in each block).*

**🪶 Cohere** — [docs.cohere.com/docs/models](https://docs.cohere.com/docs/models)
1. `command-a-plus-05-2026` — MoE flagship: vision + agentic + reasoning
2. `command-a-03-2025` — 256K context, excellent tool use / RAG, high throughput
3. `command-r-plus-08-2024` — long-lived reliable fallback

> The old `command-r-plus` / `command-r` / `command-light` aliases were
> deprecated 2025-09-15. The dated `command-r-plus-08-2024` build is still
> live and stays as the third string.

**⚡ Groq** — [console.groq.com/docs/models](https://console.groq.com/docs/models)
1. `openai/gpt-oss-120b` — featured production model, 250K TPM, tool-calling
2. `openai/gpt-oss-20b` — production, ~1 000 TPS, cheap, tool-calling
3. `qwen/qwen3.8-27b` — preview MoE, thinking + instruct modes, tool-calling

> `llama-3.1-8b-instant` **and** `llama-3.3-70b-versatile` were shut down on
> **2026-08-16** for free/developer tiers; `groq/compound*` went on
> 2026-09-21 and `qwen/qwen3.6-27b` was replaced by `qwen/qwen3.8-27b` on
> 2026-09-14. All four still appear in the picker's warning list, so a stale
> custom ID explains itself instead of 404ing silently.

**🕵️ Gemini** — [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
1. `gemini-3.8-flash` — newest stable Flash, strongest reasoning (free tier ✅)
2. `gemini-3.7-flash` — previous stable Flash (free tier ✅)
3. `gemini-3.5-flash-lite` — cheapest / fastest free-tier fallback (free tier ✅)

> Every model in this chain has a "Free of charge" Developer-API free tier
> ([pricing](https://ai.google.dev/gemini-api/docs/pricing)); Batch/Flex
> variants do **not**.

**🎛️ OpenRouter** (free-tier only by default — add paid via sidebar)
1. `z-ai/glm-4.5-air:free` — fast agent-centric MoE, hybrid thinking mode, 131K
2. `openai/gpt-oss-120b:free` — reliable general reasoning + tool calling, 131K
3. `nvidia/nemotron-3-ultra-550b-a55b:free` — 1M-token flagship MoE, tool calling
4. `meta-llama/llama-3.3-70b-instruct:free` — long-lived stable multilingual fallback

> All four were checked free & live on OpenRouter's own model pages /
> `/api/v1/models` catalogue. Free endpoints are rate-limited (20 req/min,
> 50 req/day, or 1 000 req/day after a one-time $10 credit top-up).

The catalogue fetcher now accepts any model whose **output** is text
(`…->text`), so text-out multimodal models like `qwen/qwen3.8-27b:free` and
`google/gemma-4-31b-it:free` show up in the picker too.

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
Groq/openai/gpt-oss-120b            → rate-limited
Groq/openai/gpt-oss-20b             → timeout
   [Groq exhausted, falling over to OpenRouter]
OpenRouter/z-ai/glm-4.5-air:free    → ✅ served the response
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
| `System_prompt.md` | The persona spec injected as the system message — **♂️ male variant** (he/him). Includes privacy gate, Friend Mode, family, interests, online presence. |
| `System_prompt_female.md` | **♀️ female variant** of the same persona (she/her) — the "correct gender" a new chat starts in by default. Same section structure, so the persona-mode trimming and the gender-transition event block work identically. |
| `test_tts.py` | Standalone smoke test for the free TTS engines. |
| `tests/test_fragment_policy.py` | Dependency-free static check (`python tests/test_fragment_policy.py`) that no `@st.fragment` can reach a sidebar widget — see the Development notes below. |
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

### "redacted error" / `StreamlitFragmentWidgetsNotAllowedOutsideError` on load
Streamlit Cloud redacts the message, but the real error is *"Fragments
cannot write widgets to outside containers."* It means a function wrapped
in `@st.fragment` created a widget in a container the fragment doesn't own
— almost always `st.sidebar`. The traceback's last `chatbot.py` frame names
the offending function (e.g. `_sidebar_model_settings` → `st.sidebar.button`).
Fix: remove `@st.fragment` from that function and everything it calls; only
`_render_tool_status_banner`, `_flush_lore_confirmations` and
`_maybe_show_name_popup` may be fragments. Run
`python tests/test_fragment_policy.py` to confirm before pushing.

### "rate_limit" / TPM errors from Groq
Groq's default chain (`openai/gpt-oss-120b` → `openai/gpt-oss-20b` →
`qwen/qwen3.8-27b`) sits at 250K TPM. If you added a small/legacy model via
the picker, the ~17 kB system prompt can blow its per-minute cap — remove it
or wait 60 s. Decommissioned models (both Llama IDs, `groq/compound*`,
`qwen/qwen3.6-27b`) return 404, not 429, and the sidebar lists them as
warnings if you paste one in.

### OpenRouter cost suddenly > $0.00
You added a paid model via the sidebar's "Add custom OpenRouter model"
expander. Check the `💰 Cache savings` expander's "OpenRouter spend"
line for the running session total.

### Gemini returns 404 on a preview model
The preview tier may not be enabled on your Google project. The default
chain is stable-only (`gemini-3.8-flash` → `gemini-3.7-flash` →
`gemini-3.5-flash-lite`), so the fallback loop skips a 404 and moves on —
no action needed.

### Cohere "no model selected"
The picker defaults to `command-a-plus-05-2026, command-a-03-2025,
command-r-plus-08-2024`. If you deselected all of them, the dispatcher
skips Cohere entirely on failover. Re-add at least one model in the picker.

### OpenRouter free models suddenly 429
Free endpoints are rate-limited per model (20 req/min, 50 req/day; 1 000
req/day after a one-time $10 credit top-up). Keep 3–4 free models in the
chain — the fallback walks to the next one when one is throttled.

### Gender transition button did nothing
It only fires when the target variant differs from the active one (the
button label always shows the *other* gender, so pressing it twice returns
you to where you started). Before the first message there is no context to
preserve, so the button just switches the starting variant without logging
an event. Check the `Active: ♀️/♂️` caption above it and the `↺` note under
the toggle.

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
- **`@st.fragment` may not own sidebar widgets.** A fragment can only
  create widgets inside its own container, so anything rendering into
  `st.sidebar` (the Clear Chat button, the creativity/timeout sliders, the
  provider pickers) must stay in the main script. Wrapping those in a
  fragment raises
  `StreamlitFragmentWidgetsNotAllowedOutsideError: Fragments cannot write
  widgets to outside containers.` on page load. The legal fragments in
  `chatbot.py` are `_render_tool_status_banner`,
  `_flush_lore_confirmations` and `_maybe_show_name_popup` (main-body
  renders only). `python tests/test_fragment_policy.py` fails the build if
  that rule is ever broken — directly or through a helper function.
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
