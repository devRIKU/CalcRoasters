"""Custom HTML chrome for the chat surface.

These functions render premium decorative elements that wrap around
Streamlit's native chat widgets. They use `st.markdown(..., unsafe_allow_html=True)`
rather than `components.v1.html` iframes — iframes can't reach the parent
DOM's CSS variables, and we want personality color shifts to flow through.

Public surface:
    render_header(personality, brain_type, model_id=None)
        → eyebrow tag + display title + live status pill
    render_status_dock(...)
        → optional footer dock for token count / tool activity
"""

from __future__ import annotations

import html as _html

import streamlit as st

from ui.theme import PERSONALITY_TOKENS, DEFAULT_PERSONALITY


def _eyebrow(personality: str) -> str:
    return PERSONALITY_TOKENS.get(personality, PERSONALITY_TOKENS[DEFAULT_PERSONALITY])["eyebrow"]


def render_header(
    personality: str,
    brain_type: str,
    model_id: str | None = None,
    user_name: str = "",
) -> None:
    """Premium chat header.

    Structure:
        [eyebrow pill]                           [status pill: BRAIN · MODEL · LIVE]
        Chat with Sanniva                                            (large display)
        a quiet subline that reads as intent                        (ink_dim)

    No icons, no emoji slop. The status pill on the right uses a pulsing dot
    in the personality accent — that's the only motion element on the header,
    keeping focus on the conversation below.
    """
    safe_eyebrow = _html.escape(_eyebrow(personality))
    safe_brain = _html.escape(brain_type.upper())
    safe_model = _html.escape(model_id or "")
    safe_name = _html.escape(user_name) if user_name else ""

    # Subline morphs with personality so the header feels alive on switch
    sublines = {
        "Roaster": "incoming heat. ask anything — you'll get the unfiltered cut.",
        "Smart": "considered answers. no fluff, no filler.",
        "Debater": "every position pressure-tested. bring an argument.",
        "Strategic": "options weighed, trade-offs visible. ship-ready takes.",
        "Tech Nerd": "/dev/stdin open. types, traces, and tabs all welcome.",
        "Chill Squad": "no rush. tea's brewing. ask away.",
        "Exhausted Student": "running on three cups. let's just get through this.",
    }
    subline = sublines.get(personality, "ready when you are.")

    # Personalize subline if we have a name
    if safe_name:
        subline = f"hey {safe_name} — {subline}"

    st.markdown(
        f"""
        <div class="ch-header">
          <div class="ch-header__row">
            <span class="ch-eyebrow">{safe_eyebrow}</span>
            <div class="ch-status">
              <span class="ch-status__dot"></span>
              <span class="ch-status__text">{safe_brain}</span>
              {f'<span class="ch-status__sep">·</span><span class="ch-status__model">{safe_model}</span>' if safe_model else ''}
              <span class="ch-status__sep">·</span><span class="ch-status__live">LIVE</span>
            </div>
          </div>
          <h1 class="ch-display">
            Chat with <span class="ch-display__accent">Sanniva</span><span class="ch-display__period">.</span>
          </h1>
          <p class="ch-subline">{subline}</p>
        </div>

        <style>
        .ch-header {{
            position: relative;
            margin: 0.5rem 0 2.25rem 0;
            padding: 0;
            animation: chHeaderIn 0.9s var(--ease-spring) both;
        }}
        @keyframes chHeaderIn {{
            from {{ opacity: 0; transform: translateY(20px); filter: blur(6px); }}
            to   {{ opacity: 1; transform: translateY(0);    filter: blur(0); }}
        }}
        .ch-header__row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        .ch-eyebrow {{
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.85rem;
            border: 1px solid var(--hairline-strong);
            border-radius: var(--radius-pill);
            background: rgba(255, 255, 255, 0.025);
            color: var(--ink-dim);
            font-family: var(--font-mono);
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }}
        .ch-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.3rem 0.85rem;
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid var(--hairline-strong);
            border-radius: var(--radius-pill);
            backdrop-filter: blur(10px);
            font-family: var(--font-mono);
            font-size: 0.66rem;
            letter-spacing: 0.15em;
            color: var(--ink-dim);
            text-transform: uppercase;
        }}
        .ch-status__dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 0 var(--accent);
            animation: chPulse 2s cubic-bezier(0.32, 0.72, 0, 1) infinite;
        }}
        @keyframes chPulse {{
            0%   {{ box-shadow: 0 0 0 0 var(--accent-soft); }}
            70%  {{ box-shadow: 0 0 0 8px transparent; }}
            100% {{ box-shadow: 0 0 0 0 transparent; }}
        }}
        .ch-status__text {{ color: var(--ink); font-weight: 500; }}
        .ch-status__sep   {{ opacity: 0.4; }}
        .ch-status__model {{ color: var(--ink-dim); }}
        .ch-status__live  {{ color: var(--accent); font-weight: 500; }}

        .ch-display {{
            font-family: var(--font-display) !important;
            font-size: clamp(2.5rem, 6vw, 4.25rem) !important;
            font-weight: 600 !important;
            letter-spacing: -0.035em !important;
            line-height: 0.98 !important;
            color: var(--ink) !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .ch-display__accent {{
            background: linear-gradient(180deg, var(--accent) 0%, var(--ink) 130%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent;
        }}
        .ch-display__period {{
            color: var(--accent) !important;
        }}
        .ch-subline {{
            font-family: var(--font-body) !important;
            font-size: 0.95rem !important;
            color: var(--ink-dim) !important;
            margin: 0.85rem 0 0 0 !important;
            max-width: 540px;
            line-height: 1.5;
            letter-spacing: -0.005em;
        }}

        @media (max-width: 640px) {{
            .ch-display {{ font-size: 2.25rem !important; }}
            .ch-header__row {{ flex-direction: column; align-items: flex-start; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_dock(label: str, value: str, tone: str = "default") -> None:
    """Compact inline status chip (used by tool-status banner & lore confirms).

    Args:
        label: short verb (e.g. "RUNNING", "SAVED", "FAILED")
        value: longer detail string
        tone:  "default" | "ok" | "warn" | "error" — drives the dot color
    """
    tone_color = {
        "ok": "var(--accent)",
        "warn": "#f0b649",
        "error": "#ff5a4e",
        "default": "var(--ink-dim)",
    }.get(tone, "var(--ink-dim)")

    st.markdown(
        f"""
        <div class="ch-dock">
            <span class="ch-dock__dot" style="background: {tone_color}; box-shadow: 0 0 8px {tone_color};"></span>
            <span class="ch-dock__label">{_html.escape(label)}</span>
            <span class="ch-dock__value">{_html.escape(value)}</span>
        </div>
        <style>
        .ch-dock {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.85rem;
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid var(--hairline);
            border-radius: var(--radius-pill);
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            color: var(--ink-dim);
            margin: 0.25rem 0;
            animation: chDockIn 0.5s var(--ease-spring) both;
        }}
        @keyframes chDockIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .ch-dock__dot {{
            width: 5px; height: 5px; border-radius: 50%;
        }}
        .ch-dock__label {{
            color: var(--ink); font-weight: 500; text-transform: uppercase;
        }}
        .ch-dock__value {{ opacity: 0.7; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
