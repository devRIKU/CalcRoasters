"""Premium sidebar chrome — double-bezel identity card + section dividers.

Renders the brand block + identity card at the top of the sidebar before
the native widgets (personality select, brain picker, etc.) take over.
The native widgets are restyled by ui/theme.py globally, so we don't
re-implement them here.
"""

from __future__ import annotations

import html as _html

import streamlit as st


# ---------------------------------------------------------------------------
# UI mode toggle (Classic ⇄ Ethereal Glass)
# ---------------------------------------------------------------------------

UI_MODES = ("ethereal", "classic")
UI_MODE_LABEL = {"ethereal": "Ethereal", "classic": "Classic"}


def render_mode_toggle() -> str:
    """One-click toggle between the new Ethereal Glass UI and the original
    Streamlit-native ('Classic') UI. Renders at the very top of the sidebar.

    Returns the active mode ("ethereal" | "classic"). Persists on
    `st.session_state["_ui_mode"]` so it survives reruns. Defaults to
    "ethereal" on first visit.

    Implementation notes:
        - Uses `st.segmented_control` when available (Streamlit ≥ 1.34),
          falls back to `st.radio` on older versions. Both pick the same
          session_state key so behavior is identical.
        - The label "UI" is rendered as a tiny eyebrow above the control
          so it doesn't compete with the brand block below it.
        - When the user flips Classic → Ethereal, we set a one-shot flag
          so `styles.apply_theme` knows to rewrite config.toml back to
          the Ethereal base (otherwise it'd be stuck on the last
          Classic personality's color set).
    """
    # Read current mode (with default) before the widget renders so we
    # can detect a transition AFTER the widget writes the new value.
    prev_mode = st.session_state.get("_ui_mode", "ethereal")

    # Eyebrow label + an invisible "ui-mode-toggle" CSS marker.
    # The marker is what ui/theme.py's :has() selectors anchor on to
    # find the toggle radio specifically (so we don't accidentally
    # restyle the TTS provider radio below it).
    st.sidebar.markdown(
        '<div class="ui-mode-toggle" style="font-family: ui-monospace, \'JetBrains Mono\', monospace;'
        ' font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase;'
        ' color: rgba(232,237,245,0.45); margin: 0.25rem 0 0.4rem 0;">'
        "Interface</div>",
        unsafe_allow_html=True,
    )

    # We deliberately use st.radio (not segmented_control) so the
    # CSS selectors in ui/theme.py have a single, stable target across
    # Streamlit versions. The radio gets restyled into a pill toggle.
    options = list(UI_MODES)
    mode = st.sidebar.radio(
        "UI Mode",
        options=options,
        index=options.index(prev_mode) if prev_mode in options else 0,
        format_func=lambda v: UI_MODE_LABEL[v],
        key="_ui_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    # On Classic → Ethereal transition, flag the next styles.apply_theme()
    # call to rewrite config.toml back to the Ethereal base.
    if prev_mode == "classic" and mode == "ethereal":
        st.session_state["_force_ethereal_config_write"] = True

    return mode


def render_identity_card(user_name: str = "", session_id: str = "") -> None:
    """Identity block at the top of the sidebar.

    Layout (double-bezel):
        outer shell (rounded-2xl, hairline border, p-1.5)
          inner core (rounded-xl, bg-elev-2, inner highlight)
            mark/wordmark · session id (mono)
            user_name or "Anonymous"
    """
    safe_name = _html.escape(user_name) if user_name else "Anonymous"
    safe_session = _html.escape(session_id[:8]) if session_id else "—"

    st.sidebar.markdown(
        f"""
        <div class="sb-identity">
          <div class="sb-identity__shell">
            <div class="sb-identity__core">
              <div class="sb-identity__brand-row">
                <span class="sb-identity__mark">S</span>
                <span class="sb-identity__wordmark">SANNIVA</span>
                <span class="sb-identity__build">v2.0</span>
              </div>
              <div class="sb-identity__divider"></div>
              <div class="sb-identity__name-row">
                <span class="sb-identity__name-label">Operator</span>
                <span class="sb-identity__name">{safe_name}</span>
              </div>
              <div class="sb-identity__session-row">
                <span class="sb-identity__session-label">session</span>
                <span class="sb-identity__session">{safe_session}</span>
              </div>
            </div>
          </div>
        </div>

        <style>
        .sb-identity {{
            margin: 0 0 1.5rem 0;
            animation: sbIdIn 0.7s var(--ease-spring) both;
        }}
        @keyframes sbIdIn {{
            from {{ opacity: 0; transform: translateY(-8px); filter: blur(4px); }}
            to   {{ opacity: 1; transform: translateY(0);    filter: blur(0); }}
        }}
        /* Outer shell — doppelrand */
        .sb-identity__shell {{
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid var(--hairline-strong);
            border-radius: var(--radius-card);
            padding: 6px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }}
        /* Inner core */
        .sb-identity__core {{
            background: var(--bg-elev-2);
            border-radius: var(--radius-inner);
            padding: 0.95rem 1.1rem;
            box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.06);
            position: relative;
            overflow: hidden;
        }}
        /* Subtle accent gleam in the inner core */
        .sb-identity__core::before {{
            content: "";
            position: absolute;
            top: -40%;
            right: -20%;
            width: 120px;
            height: 120px;
            background: radial-gradient(circle, var(--accent-soft) 0%, transparent 65%);
            pointer-events: none;
        }}
        .sb-identity__brand-row {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            position: relative;
            z-index: 1;
        }}
        .sb-identity__mark {{
            width: 22px; height: 22px;
            display: inline-flex; align-items: center; justify-content: center;
            background: var(--accent);
            color: #0a0a0a;
            border-radius: 6px;
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 0.72rem;
            letter-spacing: -0.02em;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.3),
                0 2px 8px var(--accent-soft);
        }}
        .sb-identity__wordmark {{
            font-family: var(--font-display);
            font-weight: 600;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            color: var(--ink);
            flex: 1;
        }}
        .sb-identity__build {{
            font-family: var(--font-mono);
            font-size: 0.6rem;
            letter-spacing: 0.05em;
            color: var(--ink-dim);
            padding: 0.15rem 0.4rem;
            border: 1px solid var(--hairline);
            border-radius: 4px;
        }}
        .sb-identity__divider {{
            height: 1px;
            background: linear-gradient(90deg,
                transparent 0%,
                var(--hairline-strong) 20%,
                var(--hairline-strong) 80%,
                transparent 100%);
            margin: 0.85rem 0;
        }}
        .sb-identity__name-row,
        .sb-identity__session-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            position: relative;
            z-index: 1;
        }}
        .sb-identity__name-row {{ margin-bottom: 0.35rem; }}
        .sb-identity__name-label,
        .sb-identity__session-label {{
            font-family: var(--font-mono);
            font-size: 0.62rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--ink-dim);
        }}
        .sb-identity__name {{
            font-family: var(--font-display);
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--ink);
            letter-spacing: -0.005em;
        }}
        .sb-identity__session {{
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--ink);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_divider(label: str) -> None:
    """Premium section divider — eyebrow tag spanning the sidebar.

    Drop this between major sidebar sections to break the visual rhythm
    away from generic stacked widgets.
    """
    safe = _html.escape(label.upper())
    st.sidebar.markdown(
        f"""
        <div class="sb-divider">
          <span class="sb-divider__line"></span>
          <span class="sb-divider__label">{safe}</span>
          <span class="sb-divider__line"></span>
        </div>
        <style>
        .sb-divider {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 1.5rem 0 0.75rem 0;
        }}
        .sb-divider__line {{
            flex: 1;
            height: 1px;
            background: var(--hairline-strong);
        }}
        .sb-divider__label {{
            font-family: var(--font-mono);
            font-size: 0.6rem;
            letter-spacing: 0.25em;
            color: var(--ink-dim);
            font-weight: 500;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
