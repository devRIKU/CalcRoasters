"""Premium dark-glass theme for the Sanniva chatbot.

Strategy: instead of rewriting `.streamlit/config.toml` per personality (which
forces a full page reload + visible flicker), we pin a single premium dark
base in config.toml and drive personality color shifts via CSS custom
properties at runtime. Result: zero-flicker personality switches.

Design language (Variance Engine roll):
    - Vibe Archetype: Ethereal Glass (OLED black + radial mesh orbs)
    - Layout Archetype: Editorial Split (sidebar identity / chat kinetic)
    - Typography: Geist (display) + Geist (body) + JetBrains Mono (HUD)
      Loaded from Google Fonts CDN. All banned fonts (Inter/Roboto/Arial)
      explicitly overridden.
    - Motion: cubic-bezier(0.32, 0.72, 0, 1) throughout. No linear/ease-in-out.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Personality color tokens
# ---------------------------------------------------------------------------
# Each personality gets:
#   accent      — primary CTA / focus ring / streaming cursor
#   accent_soft — 18% alpha version for glass tints
#   orb_a/b     — two radial gradient orbs in the ambient backdrop
#   ink         — body text color
#   ink_dim     — secondary text (timestamps, captions)
#
# All values vetted: no banned colors (no generic purple gradient slop).
# Roaster keeps the ember red identity but darkens the base so it stops
# screaming. Tech Nerd is the only one allowed to lean phosphor-green
# because that's literally the persona's signature.
PERSONALITY_TOKENS: dict[str, dict[str, str]] = {
    "Roaster": {
        "accent": "#ff5a4e",
        "accent_soft": "rgba(255, 90, 78, 0.18)",
        "orb_a": "rgba(255, 90, 78, 0.22)",
        "orb_b": "rgba(170, 30, 30, 0.16)",
        "ink": "#f5e6e2",
        "ink_dim": "rgba(245, 230, 226, 0.55)",
        "eyebrow": "Roaster · Mode 01",
    },
    "Smart": {
        "accent": "#7aa7ff",
        "accent_soft": "rgba(122, 167, 255, 0.18)",
        "orb_a": "rgba(122, 167, 255, 0.20)",
        "orb_b": "rgba(70, 110, 200, 0.14)",
        "ink": "#e8edf5",
        "ink_dim": "rgba(232, 237, 245, 0.55)",
        "eyebrow": "Smart · Mode 02",
    },
    "Debater": {
        "accent": "#f0b649",
        "accent_soft": "rgba(240, 182, 73, 0.18)",
        "orb_a": "rgba(240, 182, 73, 0.20)",
        "orb_b": "rgba(140, 90, 30, 0.16)",
        "ink": "#f0e8d8",
        "ink_dim": "rgba(240, 232, 216, 0.55)",
        "eyebrow": "Debater · Mode 03",
    },
    "Strategic": {
        "accent": "#34d399",
        "accent_soft": "rgba(52, 211, 153, 0.16)",
        "orb_a": "rgba(52, 211, 153, 0.18)",
        "orb_b": "rgba(20, 90, 70, 0.14)",
        "ink": "#dceee5",
        "ink_dim": "rgba(220, 238, 229, 0.55)",
        "eyebrow": "Strategic · Mode 04",
    },
    "Tech Nerd": {
        "accent": "#00ff9c",
        "accent_soft": "rgba(0, 255, 156, 0.14)",
        "orb_a": "rgba(0, 255, 156, 0.16)",
        "orb_b": "rgba(0, 120, 80, 0.14)",
        "ink": "#c8f5dc",
        "ink_dim": "rgba(200, 245, 220, 0.50)",
        "eyebrow": "Tech_Nerd · Mode 05",
    },
    "Chill Squad": {
        "accent": "#a4c789",
        "accent_soft": "rgba(164, 199, 137, 0.18)",
        "orb_a": "rgba(164, 199, 137, 0.20)",
        "orb_b": "rgba(110, 90, 60, 0.14)",
        "ink": "#e8e0cc",
        "ink_dim": "rgba(232, 224, 204, 0.55)",
        "eyebrow": "Chill_Squad · Mode 06",
    },
    "Exhausted Student": {
        "accent": "#9b91c4",
        "accent_soft": "rgba(155, 145, 196, 0.16)",
        "orb_a": "rgba(155, 145, 196, 0.16)",
        "orb_b": "rgba(80, 70, 110, 0.14)",
        "ink": "#cfc8e0",
        "ink_dim": "rgba(207, 200, 224, 0.50)",
        "eyebrow": "Exhausted · Mode 07",
    },
}

DEFAULT_PERSONALITY = "Smart"


def _tokens(personality: str) -> dict[str, str]:
    return PERSONALITY_TOKENS.get(personality, PERSONALITY_TOKENS[DEFAULT_PERSONALITY])


def set_personality(personality: str) -> None:
    """Persist active personality so chat_shell + sidebar can read it.

    Stored on session_state so it survives reruns within a session but
    starts fresh on a fresh tab — matches how the rest of the app
    treats per-session prefs.
    """
    st.session_state["_ui_personality"] = personality


def _fonts_block() -> str:
    """Geist + JetBrains Mono. Pinned to Google Fonts CDN.

    Geist is the Vercel display + body face — premium, geometric, distinctly
    NOT Inter. JetBrains Mono handles HUD numerics (status pill, model id).
    """
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    """


def _css(personality: str) -> str:
    t = _tokens(personality)
    # NOTE: Every rule scoped to Streamlit's actual class names. Streamlit
    # bumps these occasionally; we use the stable [data-testid] anchors
    # where they exist (st.chat_message, stChatInput, etc.).
    return f"""
    <style>
    /* --- Personality tokens (swap without page reload) ----------------- */
    :root {{
        --bg: #050505;
        --bg-elev-1: #0b0b0d;
        --bg-elev-2: #111114;
        --hairline: rgba(255, 255, 255, 0.06);
        --hairline-strong: rgba(255, 255, 255, 0.10);
        --ink: {t["ink"]};
        --ink-dim: {t["ink_dim"]};
        --accent: {t["accent"]};
        --accent-soft: {t["accent_soft"]};
        --orb-a: {t["orb_a"]};
        --orb-b: {t["orb_b"]};
        --ease-spring: cubic-bezier(0.32, 0.72, 0, 1);
        --ease-fluid: cubic-bezier(0.16, 1, 0.3, 1);
        --radius-card: 1.5rem;
        --radius-inner: calc(1.5rem - 0.375rem);
        --radius-pill: 999px;
        --font-display: "Geist", system-ui, sans-serif;
        --font-body: "Geist", system-ui, sans-serif;
        --font-mono: "JetBrains Mono", ui-monospace, monospace;
    }}

    /* --- Kill banned fonts globally ----------------------------------- */
    html, body, [class*="css"], .stApp, .stApp * {{
        font-family: var(--font-body) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    code, pre, kbd, samp, .stCode, [data-testid="stCodeBlock"] * {{
        font-family: var(--font-mono) !important;
    }}

    /* --- Ambient backdrop (radial mesh orbs) -------------------------- */
    .stApp {{
        background: var(--bg) !important;
        color: var(--ink) !important;
        position: relative;
        overflow-x: hidden;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(900px 600px at 18% 12%, var(--orb-a), transparent 60%),
            radial-gradient(700px 500px at 92% 88%, var(--orb-b), transparent 65%),
            radial-gradient(500px 400px at 55% 60%, rgba(255,255,255,0.015), transparent 70%);
        pointer-events: none;
        z-index: 0;
        transition: background 1.2s var(--ease-fluid);
    }}
    /* Film-grain overlay — fixed, pointer-events:none so it never blocks */
    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.5 0'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='0.6'/></svg>");
        opacity: 0.035;
        mix-blend-mode: overlay;
        pointer-events: none;
        z-index: 1;
    }}
    .main, [data-testid="stAppViewContainer"] > .main {{
        position: relative;
        z-index: 2;
    }}

    /* --- Top header — kill default padding + title -------------------- */
    [data-testid="stHeader"] {{
        background: transparent !important;
        backdrop-filter: blur(24px) saturate(140%);
        -webkit-backdrop-filter: blur(24px) saturate(140%);
        border-bottom: 1px solid var(--hairline);
        height: 56px;
    }}
    [data-testid="stToolbar"] {{ right: 1.5rem; }}

    /* Hide the default st.title so chat_shell can render its own header */
    .main h1:first-of-type {{
        font-family: var(--font-display) !important;
        font-size: 0;  /* hidden — chat_shell.render_header takes over */
        margin: 0;
        padding: 0;
        line-height: 0;
    }}

    /* --- Sidebar — vantablack glass card ------------------------------ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg,
            rgba(11, 11, 13, 0.92) 0%,
            rgba(7, 7, 9, 0.94) 100%) !important;
        border-right: 1px solid var(--hairline);
        backdrop-filter: blur(28px);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ display: none; }}

    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        letter-spacing: -0.015em;
        color: var(--ink) !important;
    }}
    [data-testid="stSidebar"] h3 {{
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: var(--ink-dim) !important;
        font-weight: 500 !important;
        margin-top: 1.75rem !important;
        margin-bottom: 0.5rem !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--ink) !important;
        font-size: 0.875rem;
    }}

    /* Sidebar select / text inputs — double-bezel */
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input,
    [data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {{
        background: var(--bg-elev-2) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: 0.75rem !important;
        color: var(--ink) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
        transition: border-color 0.3s var(--ease-spring), box-shadow 0.3s var(--ease-spring);
    }}
    [data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div,
    [data-testid="stSidebar"] [data-baseweb="input"]:focus-within > div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    }}

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {{
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-pill) !important;
        color: var(--ink) !important;
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.02em;
        padding: 0.5rem 1.1rem !important;
        transition: all 0.4s var(--ease-spring) !important;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: var(--accent) !important;
        transform: translateY(-1px);
    }}
    [data-testid="stSidebar"] .stButton button:active {{
        transform: scale(0.98);
    }}

    /* Sidebar info / success / error — strip default backgrounds */
    [data-testid="stSidebar"] [data-testid="stAlert"] {{
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 0.875rem !important;
        backdrop-filter: blur(8px);
    }}

    /* Sidebar slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
        background: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-soft) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {{
        background: linear-gradient(90deg, var(--accent), var(--accent-soft)) !important;
    }}

    /* Sidebar radio + checkboxes */
    [data-testid="stSidebar"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [data-baseweb="checkbox"] label {{
        color: var(--ink) !important;
        font-size: 0.85rem !important;
    }}

    /* --- Main content padding ----------------------------------------- */
    [data-testid="stAppViewContainer"] > .main > div {{
        padding-top: 1rem !important;
        max-width: 880px;
        margin: 0 auto;
    }}
    @media (max-width: 768px) {{
        [data-testid="stAppViewContainer"] > .main > div {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}

    /* --- Chat messages — double-bezel architecture -------------------- */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 1.25rem !important;
        animation: msgIn 0.7s var(--ease-spring) both;
    }}
    @keyframes msgIn {{
        from {{ opacity: 0; transform: translateY(12px); filter: blur(4px); }}
        to   {{ opacity: 1; transform: translateY(0);    filter: blur(0); }}
    }}

    /* User bubble — solid accent-tinted glass */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        display: flex;
        justify-content: flex-end;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        > div:nth-child(2) {{
        background: linear-gradient(135deg,
            var(--accent-soft) 0%,
            rgba(255, 255, 255, 0.025) 100%) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-card) !important;
        padding: 0.875rem 1.25rem !important;
        max-width: 78%;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 8px 32px rgba(0, 0, 0, 0.25);
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageAvatarUser"] {{
        order: 2;
        margin-left: 0.5rem;
        margin-right: 0;
    }}

    /* Assistant bubble — vantablack double-bezel */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        > div:nth-child(2),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"])
        > div:nth-child(2) {{
        background: var(--bg-elev-1) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: var(--radius-card) !important;
        padding: 0.875rem 1.25rem !important;
        max-width: 82%;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.04),
            0 12px 40px rgba(0, 0, 0, 0.35);
        position: relative;
    }}
    /* Inner highlight + tiny corner accent line */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        > div:nth-child(2)::before,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"])
        > div:nth-child(2)::before {{
        content: "";
        position: absolute;
        top: 0.5rem;
        left: -3px;
        width: 2px;
        height: 24px;
        background: var(--accent);
        border-radius: 2px;
        box-shadow: 0 0 12px var(--accent);
        opacity: 0.85;
    }}

    /* Avatar — clean ring */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarCustom"] {{
        background: var(--bg-elev-2) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-pill) !important;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.4);
        width: 36px !important;
        height: 36px !important;
        flex-shrink: 0;
    }}
    [data-testid="stChatMessage"] img {{
        border-radius: var(--radius-pill) !important;
    }}

    /* Chat message text */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
        color: var(--ink) !important;
        font-size: 0.9375rem !important;
        line-height: 1.65 !important;
        font-weight: 400 !important;
        text-wrap: pretty;
    }}
    [data-testid="stChatMessage"] strong {{
        color: var(--ink) !important;
        font-weight: 600 !important;
    }}
    [data-testid="stChatMessage"] code {{
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 0.375rem !important;
        padding: 0.1em 0.4em !important;
        font-size: 0.86em !important;
        color: var(--accent) !important;
    }}
    [data-testid="stChatMessage"] pre {{
        background: #06070a !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 0.875rem !important;
        padding: 1rem !important;
    }}

    /* Caption (system_note bubbles, lore saves, status banner) */
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
    [data-testid="stChatMessage"] .caption {{
        color: var(--ink-dim) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.04em;
    }}

    /* --- Chat input — fluid glass pill -------------------------------- */
    [data-testid="stChatInput"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }}
    [data-testid="stChatInputContainer"],
    [data-testid="stBottomBlockContainer"] {{
        background: transparent !important;
        border: none !important;
        padding-bottom: 1.5rem !important;
    }}
    /* Wrap the input in a glass shell */
    [data-testid="stChatInput"] > div {{
        background: rgba(11, 11, 13, 0.72) !important;
        backdrop-filter: blur(28px) saturate(180%);
        -webkit-backdrop-filter: blur(28px) saturate(180%);
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-pill) !important;
        padding: 0.25rem 0.25rem 0.25rem 1.25rem !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.05),
            0 16px 48px rgba(0, 0, 0, 0.4),
            0 0 0 0 var(--accent-soft);
        transition: box-shadow 0.5s var(--ease-spring), border-color 0.5s var(--ease-spring);
    }}
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: var(--accent) !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.05),
            0 16px 48px rgba(0, 0, 0, 0.4),
            0 0 0 4px var(--accent-soft);
    }}
    [data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: var(--ink) !important;
        font-family: var(--font-body) !important;
        font-size: 0.9375rem !important;
        font-weight: 400 !important;
        caret-color: var(--accent) !important;
        min-height: 28px !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--ink-dim) !important;
        font-style: normal;
    }}

    /* Send button — nested button-in-button pattern */
    [data-testid="stChatInput"] button {{
        background: var(--accent) !important;
        border: none !important;
        border-radius: var(--radius-pill) !important;
        width: 38px !important;
        height: 38px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        color: #0a0a0a !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.25),
            0 4px 16px var(--accent-soft);
        transition: transform 0.4s var(--ease-spring), box-shadow 0.4s var(--ease-spring);
    }}
    [data-testid="stChatInput"] button:hover {{
        transform: translateX(1px) translateY(-1px) scale(1.03);
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.25),
            0 6px 20px var(--accent-soft);
    }}
    [data-testid="stChatInput"] button:active {{
        transform: scale(0.96);
    }}
    [data-testid="stChatInput"] button svg {{
        fill: #0a0a0a !important;
        width: 16px !important;
        height: 16px !important;
    }}

    /* --- Spinner ------------------------------------------------------ */
    [data-testid="stSpinner"] > div {{
        border-color: var(--hairline-strong) !important;
        border-top-color: var(--accent) !important;
    }}
    [data-testid="stSpinner"] + div {{
        color: var(--ink-dim) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* --- Audio player ------------------------------------------------- */
    [data-testid="stAudio"] audio {{
        width: 100% !important;
        filter: hue-rotate(0deg) sepia(0.1);
        border-radius: var(--radius-pill);
    }}

    /* --- Scrollbar (webkit) ------------------------------------------- */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        border: 2px solid var(--bg);
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255, 255, 255, 0.15);
    }}

    /* --- Streamlit chrome hide --------------------------------------- */
    #MainMenu, footer, [data-testid="stStatusWidget"] {{
        visibility: hidden;
    }}
    [data-testid="stDecoration"] {{ display: none; }}

    /* --- Dialog (st.dialog for name popup) ---------------------------- */
    [data-testid="stDialog"] {{
        background: rgba(5, 5, 5, 0.85) !important;
        backdrop-filter: blur(32px);
    }}
    [data-testid="stDialog"] > div {{
        background: var(--bg-elev-1) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-card) !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.06),
            0 32px 80px rgba(0, 0, 0, 0.6);
    }}

    /* --- Expander ----------------------------------------------------- */
    [data-testid="stExpander"] {{
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 0.875rem !important;
        overflow: hidden;
    }}
    [data-testid="stExpander"] summary {{
        color: var(--ink) !important;
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }}

    /* --- UI mode toggle (segmented_control / radio fallback) ---------- */
    /* Glass-pill restyle for the sidebar Classic⇄Ethereal toggle. Targets
       both st.segmented_control (newer) and st.radio horizontal (fallback). */
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] {{
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-pill) !important;
        padding: 3px !important;
        margin-bottom: 1.25rem !important;
        backdrop-filter: blur(10px);
    }}
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label {{
        border-radius: var(--radius-pill) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.66rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.85rem !important;
        color: var(--ink-dim) !important;
        transition: all 0.4s var(--ease-spring) !important;
        background: transparent !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"]
        label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stSegmentedControl"]
        input:checked + label {{
        background: var(--bg-elev-2) !important;
        color: var(--ink) !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    /* --- Mode toggle radio (scoped to the [role=radiogroup] only) -----
       CRITICAL: we cannot blanket-restyle every [data-testid="stRadio"]
       in the sidebar — _sidebar_tts() also uses st.radio for the TTS
       provider picker. So we scope the pill restyle to the FIRST radio
       only by using :nth-of-type(1) on the wrapping stElementContainer.

       Streamlit DOM for a radio:
         <div data-testid="stRadio">
           <label data-testid="stWidgetLabel">…hidden label…</label>
           <div role="radiogroup">
             <label data-baseweb="radio">
               <div>…dot circle wrapper…</div>
               <input type="radio" value="0">
               <div>…text via stMarkdownContainer…</div>
             </label>
             …
           </div>
         </div>
       We hide the dot wrapper (div:first-child of each option label)
       and let the text container show. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.ui-mode-toggle) +
    [data-testid="stElementContainer"] [data-testid="stRadio"] > div[role="radiogroup"] {{
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid var(--hairline-strong) !important;
        border-radius: var(--radius-pill) !important;
        padding: 3px !important;
        margin-bottom: 1.25rem !important;
        gap: 0 !important;
        backdrop-filter: blur(10px);
        flex-direction: row !important;
        display: flex !important;
    }}
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.ui-mode-toggle) +
    [data-testid="stElementContainer"] [data-testid="stRadio"] label[data-baseweb="radio"] {{
        flex: 1 1 50% !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0 !important;
        border-radius: var(--radius-pill) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.66rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.6rem !important;
        color: var(--ink-dim) !important;
        transition: all 0.4s var(--ease-spring) !important;
        cursor: pointer;
        margin: 0 !important;
        background: transparent !important;
    }}
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.ui-mode-toggle) +
    [data-testid="stElementContainer"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
        background: var(--bg-elev-2) !important;
        color: var(--ink) !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    /* Hide only the dot-circle wrapper (the first <div> child of the
       option label) — NOT the markdown text container that follows. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.ui-mode-toggle) +
    [data-testid="stElementContainer"] [data-testid="stRadio"]
    label[data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}
    /* And hide the "UI Mode" widget label (we render our own eyebrow). */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.ui-mode-toggle) +
    [data-testid="stElementContainer"] [data-testid="stRadio"] > [data-testid="stWidgetLabel"] {{
        display: none !important;
    }}

    /* --- Reduced motion fallback ------------------------------------- */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    </style>
    """


def inject_theme(personality: str | None = None) -> None:
    """Inject the full premium theme stylesheet.

    Call this once per `main()` rerun, near the top, after `st.set_page_config`.
    Idempotent and cheap — Streamlit will dedupe the `<style>` block by
    content hash, so no DOM bloat across reruns.
    """
    if personality is None:
        personality = st.session_state.get("_ui_personality", DEFAULT_PERSONALITY)
    set_personality(personality)
    st.markdown(_fonts_block() + _css(personality), unsafe_allow_html=True)
