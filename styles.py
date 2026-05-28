"""Personality theme bridge — dual-mode (Classic + Ethereal Glass).

The chatbot now ships TWO UIs the user can toggle between:

- **Classic** — the original Streamlit-native chrome. Themes by rewriting
  `.streamlit/config.toml` per personality (causes a page reload, but
  preserves the original behavior 1:1 for users who prefer it).

- **Ethereal Glass** — the new premium custom UI. Themes via CSS variables
  injected by `ui/theme.py`. Zero page reload, zero flicker.

`apply_theme()` reads `st.session_state["_ui_mode"]` to decide which path
to take. Default mode is "ethereal" so first-time users land on the new UI.
"""

from __future__ import annotations

import os

try:
    import streamlit as st  # type: ignore
except ImportError:  # pragma: no cover - test harnesses
    st = None  # type: ignore

import toml


# Per-personality color tokens for the CLASSIC mode (config.toml rewrite path).
# These match the original repo behavior exactly — do NOT edit without also
# updating ui/theme.py PERSONALITY_TOKENS for parity.
_CLASSIC_THEMES: dict[str, dict[str, str]] = {
    "Roaster": {
        "primaryColor": "#ff4444",
        "backgroundColor": "#1a0505",
        "secondaryBackgroundColor": "#2e0a0a",
        "textColor": "#ffcccc",
    },
    "Smart": {
        "primaryColor": "#4299e1",
        "backgroundColor": "#f0f4f8",
        "secondaryBackgroundColor": "#ffffff",
        "textColor": "#1a202c",
    },
    "Debater": {
        "primaryColor": "#d69e2e",
        "backgroundColor": "#2d3748",
        "secondaryBackgroundColor": "#1a202c",
        "textColor": "#e2e8f0",
    },
    "Strategic": {
        "primaryColor": "#10b981",
        "backgroundColor": "#0f172a",
        "secondaryBackgroundColor": "#1e293b",
        "textColor": "#cbd5e1",
    },
    "Tech Nerd": {
        "primaryColor": "#00ff9c",
        "backgroundColor": "#0a0f0d",
        "secondaryBackgroundColor": "#11181a",
        "textColor": "#b8f5d6",
    },
    "Chill Squad": {
        "primaryColor": "#7a9e6e",
        "backgroundColor": "#f5efe2",
        "secondaryBackgroundColor": "#e8e0cc",
        "textColor": "#3d3a2f",
    },
    "Exhausted Student": {
        "primaryColor": "#6b6488",
        "backgroundColor": "#1c1a26",
        "secondaryBackgroundColor": "#262332",
        "textColor": "#8e88a3",
    },
}

# Pinned Ethereal Glass base — written once when the user is in Ethereal
# mode, never per-personality (personality color shift happens via CSS
# variables at runtime instead). Matches `.streamlit/config.toml`.
_ETHEREAL_BASE: dict[str, str] = {
    "base": "dark",
    "primaryColor": "#7AA7FF",
    "backgroundColor": "#050505",
    "secondaryBackgroundColor": "#0b0b0d",
    "textColor": "#e8edf5",
}


def _write_config(theme_block: dict[str, str]) -> None:
    """Write the [theme] section of `.streamlit/config.toml` only if the
    target values differ from the on-disk values. Avoids needless reloads.
    """
    config_path = ".streamlit/config.toml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    current_config: dict = {}
    if os.path.exists(config_path):
        try:
            current_config = toml.load(config_path)
        except Exception:
            current_config = {}

    current_config.setdefault("theme", {})
    needs_update = False
    for key, value in theme_block.items():
        if current_config["theme"].get(key) != value:
            current_config["theme"][key] = value
            needs_update = True

    if needs_update:
        with open(config_path, "w") as f:
            toml.dump(current_config, f)


def apply_theme(personality: str) -> None:
    """Dispatch theme application based on the active UI mode.

    - `_ui_mode == "classic"` → rewrite `.streamlit/config.toml` with the
      personality's color set (original behavior).
    - `_ui_mode == "ethereal"` (default) → just persist the personality
      on session_state; `ui.theme.inject_theme()` will read it and swap
      CSS variables on the next render. config.toml is left at the
      pinned Ethereal Glass base.
    """
    if st is None:
        return

    # Always persist the personality — the Ethereal CSS injector reads it
    # and the Classic path uses it for the theme lookup below.
    st.session_state["_ui_personality"] = personality

    mode = st.session_state.get("_ui_mode", "ethereal")

    if mode == "classic":
        theme = _CLASSIC_THEMES.get(personality)
        if theme:
            _write_config(theme)
    else:
        # Ethereal — make sure the base is pinned (no per-personality rewrite).
        # We DO NOT call this on every rerun: only when the user actively
        # switches into Ethereal from Classic, signaled by a transient flag.
        if st.session_state.pop("_force_ethereal_config_write", False):
            _write_config(_ETHEREAL_BASE)
