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
import sqlite3
import threading
import time
from typing import Any

LORE_FILE = os.path.join(os.path.dirname(__file__), "lore.json")
DB_FILE = os.path.join(os.path.dirname(__file__), "private_lore.db")
_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None


def _init_db() -> None:
    with _lock:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS private_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_key TEXT,
                    fact TEXT,
                    ts INTEGER
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_key ON private_facts(user_key);")


# Initialize database
_init_db()


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
    """Append a fact to a user's lore in the private SQLite database. Registers the user profile in lore.json if missing."""
    if not name or not fact or not fact.strip():
        return {"ok": False, "error": "name and fact required"}

    # Ensure user profile exists in public lore (lore.json)
    ensure_user(name)
    k = _key(name)
    now = int(time.time())

    with sqlite3.connect(DB_FILE) as conn:
        # Check for duplicate in private DB
        cursor = conn.execute(
            "SELECT 1 FROM private_facts WHERE user_key = ? AND LOWER(TRIM(fact)) = LOWER(TRIM(?)) LIMIT 1",
            (k, fact)
        )
        if cursor.fetchone():
            return {"ok": True, "duplicate": True}

        # Check for duplicate in public lore.json to avoid saving redundant facts
        public_facts = {f.lower().strip() for f in list_public_facts(name)}
        if fact.strip().lower() in public_facts:
            return {"ok": True, "duplicate": True}

        # Insert new fact
        conn.execute(
            "INSERT INTO private_facts (user_key, fact, ts) VALUES (?, ?, ?)",
            (k, fact.strip(), now)
        )
    return {"ok": True, "duplicate": False}


def list_public_facts(name: str) -> list[str]:
    """Return list of public fact strings from lore.json for a user (most recent first)."""
    rec = get_user(name)
    if not rec:
        return []
    facts = rec.get("facts", [])
    return [f.get("text", "") for f in reversed(facts) if f.get("text")]


def list_private_facts(name: str) -> list[str]:
    """Return list of private fact strings from the SQLite database for a user (most recent first)."""
    k = _key(name)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT fact FROM private_facts WHERE user_key = ? ORDER BY id DESC",
            (k,)
        )
        return [row[0] for row in cursor.fetchall()]


def list_facts(name: str) -> list[str]:
    """Alias for list_public_facts. Used by Streamlit UI to display public lore only."""
    return list_public_facts(name)


def remove_fact(name: str, fact_substring: str) -> dict[str, Any]:
    """Remove facts whose text contains the substring (case-insensitive) from both public and private databases."""
    removed_public = 0
    removed_private = 0
    k = _key(name)
    needle = fact_substring.strip().lower()

    # Remove from lore.json (public)
    with _lock:
        db = _load()
        if k in db["users"]:
            before = len(db["users"][k]["facts"])
            db["users"][k]["facts"] = [
                f for f in db["users"][k]["facts"]
                if needle not in f.get("text", "").lower()
            ]
            removed_public = before - len(db["users"][k]["facts"])
            if removed_public > 0:
                _save(db)

    # Remove from SQLite (private)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM private_facts WHERE user_key = ? AND LOWER(fact) LIKE ?",
            (k, f"%{needle}%")
        )
        removed_private = cursor.fetchone()[0]
        if removed_private > 0:
            conn.execute(
                "DELETE FROM private_facts WHERE user_key = ? AND LOWER(fact) LIKE ?",
                (k, f"%{needle}%")
            )

    return {"ok": True, "removed_public": removed_public, "removed_private": removed_private}


def all_users() -> list[str]:
    """Return list of known display names."""
    with _lock:
        db = _load()
        return [u.get("name", k) for k, u in db["users"].items()]


def render_lore_block(name: str, max_facts: int = 20) -> str:
    """Build a markdown snippet containing both public and private facts to inject into a system prompt."""
    public_facts = list_public_facts(name)
    private_facts = list_private_facts(name)
    all_facts = public_facts + private_facts
    if not all_facts:
        return ""

    display_name = name.strip()
    rec = get_user(name)
    if rec:
        display_name = rec.get("name", display_name)

    lines = [f"## Known facts about {display_name} (from past chats)"]
    for f in all_facts[:max_facts]:
        lines.append(f"- {f}")
    return "\n".join(lines)
