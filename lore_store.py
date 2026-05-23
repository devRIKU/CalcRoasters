"""
Lore Store
==========
Persistent per-user memory for Sanniva. Stores facts/lore the chatbot
learns or is told about a given user, keyed by their name.

Storage format (lore.json):
{
    "users": {
        "ayushi": {
            "name": "Ayushi",
            "facts": [
                {"text": "Loves Demon Slayer", "ts": 1690000000},
                ...
            ],
            "first_seen": 1690000000,
            "last_seen": 1690000000
        },
        ...
    }
}
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

LORE_FILE = os.path.join(os.path.dirname(__file__), "lore.json")
_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None


def _empty_db() -> dict[str, Any]:
    return {"users": {}}


def _load() -> dict[str, Any]:
    global _cache, _cache_mtime
    if not os.path.exists(LORE_FILE):
        _cache = _empty_db()
        _cache_mtime = None
        return _empty_db()
    try:
        mtime = os.path.getmtime(LORE_FILE)
        if _cache is not None and _cache_mtime == mtime:
            return _cache
        with open(LORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "users" not in data:
            return _empty_db()
        _cache = data
        _cache_mtime = mtime
        return data
    except Exception:
        return _empty_db()


def _save(db: dict[str, Any]) -> None:
    global _cache, _cache_mtime
    tmp = LORE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, LORE_FILE)
    _cache = db
    try:
        _cache_mtime = os.path.getmtime(LORE_FILE)
    except Exception:
        _cache_mtime = None


def _key(name: str) -> str:
    return (name or "").strip().lower()


def get_user(name: str) -> dict[str, Any] | None:
    """Return the stored record for `name` or None."""
    if not name:
        return None
    with _lock:
        db = _load()
        return db["users"].get(_key(name))


def ensure_user(name: str) -> dict[str, Any]:
    """Create the user record if missing; return it."""
    with _lock:
        db = _load()
        k = _key(name)
        if k not in db["users"]:
            now = int(time.time())
            db["users"][k] = {
                "name": name.strip(),
                "facts": [],
                "first_seen": now,
                "last_seen": now,
            }
            _save(db)
        return db["users"][k]


def add_fact(name: str, fact: str) -> dict[str, Any]:
    """Append a fact to a user's lore. Creates the user if necessary."""
    if not name or not fact or not fact.strip():
        return {"ok": False, "error": "name and fact required"}
    with _lock:
        db = _load()
        k = _key(name)
        if k not in db["users"]:
            now = int(time.time())
            db["users"][k] = {
                "name": name.strip(),
                "facts": [],
                "first_seen": now,
                "last_seen": now,
            }
        # de-dupe: skip if exact text already present
        existing = {f.get("text", "").strip().lower() for f in db["users"][k]["facts"]}
        if fact.strip().lower() in existing:
            return {"ok": True, "duplicate": True, "user": db["users"][k]}
        db["users"][k]["facts"].append({"text": fact.strip(), "ts": int(time.time())})
        db["users"][k]["last_seen"] = int(time.time())
        _save(db)
        return {"ok": True, "duplicate": False, "user": db["users"][k]}


def list_facts(name: str) -> list[str]:
    """Return list of fact strings for a user (most recent first)."""
    rec = get_user(name)
    if not rec:
        return []
    facts = rec.get("facts", [])
    return [f.get("text", "") for f in reversed(facts) if f.get("text")]


def remove_fact(name: str, fact_substring: str) -> dict[str, Any]:
    """Remove facts whose text contains the substring (case-insensitive)."""
    with _lock:
        db = _load()
        k = _key(name)
        if k not in db["users"]:
            return {"ok": False, "error": "user not found"}
        needle = fact_substring.strip().lower()
        before = len(db["users"][k]["facts"])
        db["users"][k]["facts"] = [
            f for f in db["users"][k]["facts"]
            if needle not in f.get("text", "").lower()
        ]
        removed = before - len(db["users"][k]["facts"])
        _save(db)
        return {"ok": True, "removed": removed}


def all_users() -> list[str]:
    """Return list of known display names."""
    with _lock:
        db = _load()
        return [u.get("name", k) for k, u in db["users"].items()]


def render_lore_block(name: str, max_facts: int = 20) -> str:
    """Build a markdown snippet to inject into a system prompt."""
    rec = get_user(name)
    if not rec:
        return ""
    facts = rec.get("facts", [])[-max_facts:]
    if not facts:
        return ""
    lines = [f"## Known facts about {rec.get('name', name)} (from past chats)"]
    for f in facts:
        lines.append(f"- {f.get('text', '')}")
    return "\n".join(lines)
