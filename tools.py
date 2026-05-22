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

import json
from typing import Any

import streamlit as st

import lore_store


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
                "etc.). Keep each fact short and concrete."
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
    """Run a tool and return a JSON-serialisable result."""
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
            result = lore_store.add_fact(user_name, fact)
            return {
                "ok": result.get("ok", False),
                "duplicate": result.get("duplicate", False),
                "user_name": user_name,
                "fact": fact,
            }

        if name == "recall_lore":
            user_name = (args.get("user_name") or "").strip()
            facts = lore_store.list_facts(user_name)
            return {"ok": True, "user_name": user_name, "facts": facts}

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
