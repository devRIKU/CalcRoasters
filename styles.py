"""
Theme handling — optimized for smoothness
==========================================
Old version rewrote .streamlit/config.toml on every personality switch,
causing Streamlit to detect a config change and trigger a full reload.
That made the app feel laggy.

New version injects CSS variables directly via st.markdown — instant,
no file I/O, no reload, no disk thrash. We still optionally update the
config.toml file BUT only if explicitly requested (e.g. for persistence),
and we debounce it.

Also caches theme lookups and avoids redundant injections.
"""

from __future__ import annotations

import os
from typing import Dict

import streamlit as st

# Define themes — same palette as before, but now applied via CSS
_THEMES: Dict[str, Dict[str, str]] = {
    "Roaster": {
        "primaryColor": "#ff4444",
        "backgroundColor": "#1a0505",
        "secondaryBackgroundColor": "#2e0a0a",
        "textColor": "#ffcccc",
        "accent": "#ff6b6b",
        "border": "#4a1a1a",
    },
    "Smart": {
        "primaryColor": "#4299e1",
        "backgroundColor": "#f0f4f8",
        "secondaryBackgroundColor": "#ffffff",
        "textColor": "#1a202c",
        "accent": "#63b3ed",
        "border": "#bee3f8",
    },
    "Debater": {
        "primaryColor": "#d69e2e",
        "backgroundColor": "#2d3748",
        "secondaryBackgroundColor": "#1a202c",
        "textColor": "#e2e8f0",
        "accent": "#ecc94b",
        "border": "#4a5568",
    },
    "Strategic": {
        "primaryColor": "#10b981",
        "backgroundColor": "#0f172a",
        "secondaryBackgroundColor": "#1e293b",
        "textColor": "#cbd5e1",
        "accent": "#34d399",
        "border": "#334155",
    },
    "Tech Nerd": {
        "primaryColor": "#00ff9c",
        "backgroundColor": "#0a0f0d",
        "secondaryBackgroundColor": "#11181a",
        "textColor": "#b8f5d6",
        "accent": "#00ff9c",
        "border": "#1a2e26",
    },
    "Chill Squad": {
        "primaryColor": "#7a9e6e",
        "backgroundColor": "#f5efe2",
        "secondaryBackgroundColor": "#e8e0cc",
        "textColor": "#3d3a2f",
        "accent": "#a3c49a",
        "border": "#d6cfb8",
    },
    "Exhausted Student": {
        "primaryColor": "#6b6488",
        "backgroundColor": "#1c1a26",
        "secondaryBackgroundColor": "#262332",
        "textColor": "#8e88a3",
        "accent": "#9a95b0",
        "border": "#3a3650",
    },
}

# Track last applied theme to avoid redundant CSS injection
_last_theme: str | None = None

def get_theme(personality: str) -> Dict[str, str]:
    return _THEMES.get(personality, _THEMES["Roaster"])

def _build_css(theme: Dict[str, str], personality: str) -> str:
    """Build minimal CSS that themes Streamlit without full reload."""
    bg = theme["backgroundColor"]
    secondary = theme["secondaryBackgroundColor"]
    text = theme["textColor"]
    primary = theme["primaryColor"]
    accent = theme.get("accent", primary)
    border = theme.get("border", secondary)

    # We target Streamlit's main containers. Use subtle theming — not
    # aggressive overrides that break Streamlit's own components.
    return f"""
<style>
/* Theme: {personality} — injected via styles.py (no config.toml write) */
:root {{
    --sanniva-primary: {primary};
    --sanniva-bg: {bg};
    --sanniva-secondary: {secondary};
    --sanniva-text: {text};
    --sanniva-accent: {accent};
    --sanniva-border: {border};
}}

/* Main app background — subtle tint, not full override to keep readability */
.stApp {{
    background-color: {bg} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {secondary} !important;
    border-right: 1px solid {border};
}}
section[data-testid="stSidebar"] * {{
    color: {text} !important;
}}

/* Chat messages — tint assistant bubbles */
div[data-testid="stChatMessage"] {{
    border: 1px solid {border};
    border-radius: 12px;
    margin-bottom: 8px;
}}

/* Buttons — primary color */
.stButton > button {{
    background-color: {primary} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background-color: {accent} !important;
    transform: translateY(-1px);
}}

/* Selectbox / inputs — subtle border */
div[data-baseweb="select"], div[data-testid="stTextInput"] input {{
    border-color: {border} !important;
}}

/* Personality-specific flourishes */
{"/* Roaster: subtle glow on primary */" if personality == "Roaster" else ""}
{".stApp { box-shadow: inset 0 0 100px rgba(255,68,68,0.05); }" if personality == "Roaster" else ""}
{".stApp { box-shadow: inset 0 0 100px rgba(0,255,156,0.05); }" if personality == "Tech Nerd" else ""}
{".stApp { box-shadow: inset 0 0 80px rgba(122,158,110,0.08); }" if personality == "Chill Squad" else ""}

/* Smooth transitions for theme changes */
.stApp, section[data-testid="stSidebar"], .stButton > button {{
    transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
}}
</style>
"""

def apply_theme(personality: str, *, force: bool = False, persist_to_file: bool = False) -> None:
    """
    Apply theme for personality.

    - By default, injects CSS instantly (no file I/O, no reload).
    - If persist_to_file=True, also writes .streamlit/config.toml (debounced,
      only if changed). This is opt-in now — old behavior was always persisting,
      which caused lag.

    Args:
        personality: one of the 7 modes
        force: re-inject even if same as last time
        persist_to_file: whether to also write config.toml for persistence across restarts
    """
    global _last_theme

    if not force and _last_theme == personality:
        # Already applied this session, skip injection
        return

    theme = get_theme(personality)
    css = _build_css(theme, personality)

    try:
        # Inject CSS — this is instant, no reload
        st.markdown(css, unsafe_allow_html=True)
        _last_theme = personality
    except Exception:
        # If st.markdown fails (e.g. outside Streamlit context in tests), silently ignore
        pass

    # Optional persistence — only if explicitly requested
    if persist_to_file:
        _persist_theme_to_file(theme)

def _persist_theme_to_file(theme: Dict[str, str]) -> None:
    """Legacy file persistence — now opt-in and debounced."""
    try:
        import toml
        config_path = ".streamlit/config.toml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        current_config = {}
        if os.path.exists(config_path):
            try:
                current_config = toml.load(config_path)
            except Exception:
                pass

        if "theme" not in current_config:
            current_config["theme"] = {}

        needs_update = False
        for key in ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"):
            if current_config["theme"].get(key) != theme.get(key):
                current_config["theme"][key] = theme.get(key)
                needs_update = True

        if needs_update:
            with open(config_path, "w") as f:
                toml.dump(current_config, f)
    except Exception:
        pass

def clear_theme_cache():
    global _last_theme
    _last_theme = None
