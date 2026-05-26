"""
AI Tool Definitions
===================
Function-calling tools the chatbot brain can invoke.

We define them in two forms:
- OpenAI/Groq spec (JSON-schema, "type": "function").
- Gemini spec (google.genai.types.Tool / FunctionDeclaration).

A central `dispatch(name, args)` runs the side-effects (lore storage,
queuing a Streamlit popup, etc) and returns a JSON-serialisable result
for the model.
"""
from __future__ import annotations

import concurrent.futures
import json
import queue
import time
from typing import Any, Iterable

import streamlit as st

import lore_store


# ---------------------------------------------------------------------------
# Thread pool for parallel tool dispatch
# ---------------------------------------------------------------------------
# A single shared pool keeps the cost of repeated tool calls low (no per-call
# thread spin-up) and bounds concurrency so we don't open 50 Firestore
# connections if the model goes wild.
_TOOL_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="sanniva-tool",
)


def _attach_streamlit_context(fn):
    """Wrap `fn` so it runs inside the current Streamlit ScriptRunContext.

    Tools touch `st.session_state` (e.g. _lore_save_confirmations), and
    background threads in Streamlit have no script context by default —
    they'd hit `Missing ScriptRunContext` warnings and `session_state`
    writes wouldn't reach the active session. `add_script_run_ctx` is the
    documented escape hatch.

    Falls back to running `fn` unchanged if the helper isn't available
    (older Streamlit or non-Streamlit test harnesses).
    """
    try:
        from streamlit.runtime.scriptrunner import (  # type: ignore
            add_script_run_ctx, get_script_run_ctx,
        )
        ctx = get_script_run_ctx()
    except Exception:
        return fn

    if ctx is None:
        return fn

    def _wrapped(*a, **kw):
        try:
            import threading
            add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:
            pass
        return fn(*a, **kw)

    return _wrapped


# ---------------------------------------------------------------------------
# Cross-thread status bus
# ---------------------------------------------------------------------------
# A standard library `queue.Queue` lives on session_state. Tool dispatch
# (whether sync on the script thread or off-thread in a future ThreadPool)
# pushes lightweight status dicts onto it; the Streamlit render loop drains
# them on the next paint via `_drain_tool_status_queue` in chatbot.py.
#
# This is the pattern Streamlit officially recommends because background
# threads cannot safely touch session_state or call st.* — they can only
# enqueue messages. Even though `dispatch()` currently runs synchronously,
# routing status through the queue means we can later move slow tools onto
# a ThreadPoolExecutor without rewriting the UI.

def _status_queue() -> "queue.Queue[dict[str, Any]]":
    """Return (and lazily create) the session-scoped status queue."""
    q = st.session_state.get("_tool_status_queue")
    if q is None:
        q = queue.Queue()
        st.session_state["_tool_status_queue"] = q
    return q


def push_tool_status(event: str, **fields: Any) -> None:
    """Push a status event onto the cross-thread queue.

    `event` is one of: 'started', 'progress', 'finished', 'error'.
    Extra fields (tool name, args summary, error string) ride along.
    """
    try:
        _status_queue().put_nowait({"event": event, "ts": time.time(), **fields})
    except Exception:
        # Never let status reporting take down a tool dispatch.
        pass


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Groq style)
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "request_user_name",
            "description": (
                "Open a popup asking the user to provide their name when you "
                "don't know it yet. Use this the FIRST time you talk to someone "
                "you don't recognise so you can store lore about them later. "
                "Do not call it again if the name is already known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short, friendly reason shown in the popup.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_lore",
            "description": (
                "Save a fact about a specific user so you can recall it in future "
                "conversations. Call this whenever the user shares something "
                "memorable about themselves (likes, dislikes, family, hobbies, "
                "etc.). Keep each fact short and concrete.\n\n"
                "YOU MUST decide whether the fact is sensitive and set the "
                "`private` flag accordingly:\n"
                "  • `private: true`  → sensitive / personal info that should NOT "
                "be visible to other users of this app. Examples: real address, "
                "phone, email, school name combined with grade+location, mental "
                "health, family conflict, religion, sexuality, romantic interests, "
                "exam scores, salary, anything the user said 'don't tell anyone' "
                "about. Saved to Firebase (or a local encrypted-style SQLite "
                "fallback) — not shown in any public listing.\n"
                "  • `private: false` → harmless preferences and trivia that are "
                "fine to share publicly. Examples: favourite anime, favourite "
                "food, hobbies, pets' names, favourite subject, music taste, "
                "the fact that they own a bike. Saved to public lore.json.\n"
                "When in doubt, prefer `private: true`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "type": "string",
                        "description": "Display name of the user the fact is about.",
                    },
                    "fact": {
                        "type": "string",
                        "description": "Single concrete fact, e.g. 'Loves Demon Slayer'.",
                    },
                    "private": {
                        "type": "boolean",
                        "description": (
                            "True if the fact is sensitive/personal (Firebase / "
                            "private store). False if it's a harmless preference "
                            "(public lore.json). Default: true (err on the side "
                            "of privacy)."
                        ),
                    },
                },
                "required": ["user_name", "fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_lore",
            "description": (
                "Look up everything you remember about a user by name. "
                "Returns a list of stored facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "type": "string",
                        "description": "Display name of the user to look up.",
                    }
                },
                "required": ["user_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Gemini tool builder
# ---------------------------------------------------------------------------

def build_gemini_tools():
    """Return a list[types.Tool] for Gemini, or [] if SDK unavailable."""
    try:
        from google.genai import types  # type: ignore
    except Exception:
        return []

    decls = []
    for t in OPENAI_TOOLS:
        f = t["function"]
        decls.append(
            types.FunctionDeclaration(
                name=f["name"],
                description=f["description"],
                parameters=f["parameters"],
            )
        )
    return [types.Tool(function_declarations=decls)]


# ---------------------------------------------------------------------------
# Dispatcher — actually executes the tool side effects
# ---------------------------------------------------------------------------

def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool and return a JSON-serialisable result.

    Emits a status event onto the cross-thread queue before and after each
    tool runs so the UI can render an optimistic "🔧 running X…" placeholder
    without blocking on the actual tool work.
    """
    push_tool_status("started", tool=name, args=_summarise_args(args))
    started_at = time.time()
    try:
        result = _dispatch_inner(name, args)
        push_tool_status(
            "finished",
            tool=name,
            duration_ms=int((time.time() - started_at) * 1000),
            ok=bool(result.get("ok", True)),
        )
        return result
    except Exception as e:
        push_tool_status(
            "error",
            tool=name,
            duration_ms=int((time.time() - started_at) * 1000),
            error=f"{type(e).__name__}: {e}",
        )
        return {"ok": False, "error": str(e)}


def _summarise_args(args: dict[str, Any], max_len: int = 60) -> str:
    """Build a short human-readable summary of tool args for the status bus.

    Avoid leaking long lore facts into the queue — truncate to `max_len`.
    """
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_len:
            v = v[:max_len] + "…"
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _dispatch_inner(name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "request_user_name":
            # Signal to the Streamlit layer that a popup is needed.
            st.session_state["_name_popup_pending"] = True
            st.session_state["_name_popup_reason"] = args.get("reason") or ""
            return {
                "status": "popup_opened",
                "note": "A name input popup is being shown to the user. "
                        "Continue your reply naturally; the name will be "
                        "available on the next turn.",
            }

        if name == "remember_lore":
            user_name = (args.get("user_name") or "").strip()
            fact = (args.get("fact") or "").strip()
            if not user_name or not fact:
                return {"ok": False, "error": "user_name and fact required"}
            # Default to private when unspecified — safer than leaking sensitive
            # info into public lore.json because the model forgot the flag.
            private_raw = args.get("private")
            if isinstance(private_raw, str):
                private = private_raw.strip().lower() in ("true", "1", "yes", "y")
            elif private_raw is None:
                private = True
            else:
                private = bool(private_raw)

            result = lore_store.add_fact(user_name, fact, private=private)
            visibility = result.get("visibility", "private" if private else "public")
            backend = result.get("backend", "")

            # Queue an in-chat confirmation banner. The Streamlit layer drains
            # this queue at the top of each render and prints the messages
            # inline in the transcript — NOT as a popup/toast.
            if result.get("ok") and not result.get("duplicate"):
                pending = st.session_state.setdefault("_lore_save_confirmations", [])
                pending.append({
                    "user_name": user_name,
                    "fact": fact,
                    "visibility": visibility,
                    "backend": backend,
                })

            return {
                "ok": result.get("ok", False),
                "duplicate": result.get("duplicate", False),
                "visibility": visibility,
                "backend": backend,
                "user_name": user_name,
                "fact": fact,
            }

        if name == "recall_lore":
            user_name = (args.get("user_name") or "").strip()
            public = lore_store.list_public_facts(user_name)
            private = lore_store.list_private_facts(user_name)
            return {
                "ok": True,
                "user_name": user_name,
                "facts": public + private,       # combined for convenience
                "public_facts": public,
                "private_facts": private,
            }

        return {"ok": False, "error": f"unknown tool: {name}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def dispatch_json(name: str, args_json: str) -> str:
    """Convenience: take/return JSON strings (for Groq tool responses)."""
    try:
        args = json.loads(args_json) if args_json else {}
    except Exception:
        args = {}
    return json.dumps(dispatch(name, args), ensure_ascii=False)


def dispatch_parallel(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run multiple tool calls concurrently on the shared thread pool.

    `calls` is an iterable of dicts shaped like the normalised tool-call
    record used by the brain loops::

        [{"id": "...", "name": "remember_lore",
          "arguments": '{"user_name":"X","fact":"Y","private":true}'}, ...]

    Returns the SAME list shape with `result` (the dict from dispatch) and
    `content` (the JSON-serialised result, for direct injection into a
    Groq `role: "tool"` message) filled in. Order matches the input.

    Tools that don't share state (which is the case here — remember_lore
    serialises lore_store internally with a Lock) are safe to run in
    parallel. The combined wall time is bounded by the slowest tool, not
    the sum.
    """
    call_list = list(calls)
    if not call_list:
        return []

    # Snapshot the current ScriptRunContext once so every worker inherits it.
    ctx = None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        ctx = get_script_run_ctx()
    except Exception:
        pass

    def _run_one(tc: dict[str, Any]) -> dict[str, Any]:
        if ctx is not None:
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore
                import threading
                add_script_run_ctx(threading.current_thread(), ctx)
            except Exception:
                pass
        try:
            args = json.loads(tc.get("arguments") or "{}")
        except Exception:
            args = {}
        result = dispatch(tc["name"], args)
        return {
            **tc,
            "result": result,
            "content": json.dumps(result, ensure_ascii=False),
        }

    # Use the shared pool's map() so we don't spin up new threads each call.
    futures = [_TOOL_POOL.submit(_run_one, tc) for tc in call_list]
    return [f.result() for f in futures]
