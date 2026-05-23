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

_use_firebase = False
_firestore_db = None


def _init_firebase() -> None:
    global _use_firebase, _firestore_db
    firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    firebase_project_id = os.environ.get("FIREBASE_PROJECT_ID")
    firebase_private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
    firebase_client_email = os.environ.get("FIREBASE_CLIENT_EMAIL")

    if firebase_creds_json or (firebase_project_id and firebase_private_key and firebase_client_email):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            if not firebase_admin._apps:
                if firebase_creds_json:
                    if firebase_creds_json.strip().startswith("{"):
                        cred_dict = json.loads(firebase_creds_json)
                        cred = credentials.Certificate(cred_dict)
                    else:
                        cred = credentials.Certificate(firebase_creds_json)
                else:
                    private_key = firebase_private_key.replace("\\n", "\n")
                    cred_dict = {
                        "type": "service_account",
                        "project_id": firebase_project_id,
                        "private_key": private_key,
                        "client_email": firebase_client_email,
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                    cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            _firestore_db = firestore.client()
            _use_firebase = True
        except Exception as e:
            import sys
            sys.stderr.write(f"Firebase initialization failed: {e}. Using SQLite fallback.\n")


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


# Initialize datastores
_init_firebase()
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
    """Append a fact to a user's lore in the private database (Firestore or SQLite fallback). Registers the user profile in lore.json if missing."""
    if not name or not fact or not fact.strip():
        return {"ok": False, "error": "name and fact required"}

    # Ensure user profile exists in public lore (lore.json)
    ensure_user(name)
    k = _key(name)
    now = int(time.time())
    fact_clean = fact.strip()

    # Check for duplicate in public lore.json to avoid saving redundant facts
    public_facts = {f.lower().strip() for f in list_public_facts(name)}
    if fact_clean.lower() in public_facts:
        return {"ok": True, "duplicate": True}

    if _use_firebase and _firestore_db is not None:
        try:
            # Query all facts for user to check duplicate in Firestore
            docs = _firestore_db.collection("private_facts").where("user_key", "==", k).stream()
            for doc in docs:
                if doc.to_dict().get("fact", "").strip().lower() == fact_clean.lower():
                    return {"ok": True, "duplicate": True}
            
            # Insert new fact to Firebase Firestore
            _firestore_db.collection("private_facts").add({
                "user_key": k,
                "fact": fact_clean,
                "ts": now
            })
            return {"ok": True, "duplicate": False}
        except Exception as e:
            import sys
            sys.stderr.write(f"Firestore add_fact failed: {e}. Falling back to SQLite.\n")

    # SQLite Fallback
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM private_facts WHERE user_key = ? AND LOWER(TRIM(fact)) = LOWER(TRIM(?)) LIMIT 1",
            (k, fact_clean)
        )
        if cursor.fetchone():
            return {"ok": True, "duplicate": True}

        conn.execute(
            "INSERT INTO private_facts (user_key, fact, ts) VALUES (?, ?, ?)",
            (k, fact_clean, now)
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
    """Return list of private fact strings from the database (Firestore or SQLite fallback) for a user (most recent first)."""
    k = _key(name)

    if _use_firebase and _firestore_db is not None:
        try:
            docs = _firestore_db.collection("private_facts").where("user_key", "==", k).stream()
            facts_list = []
            for doc in docs:
                d = doc.to_dict()
                facts_list.append((d.get("fact", ""), d.get("ts", 0)))
            facts_list.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in facts_list if f[0]]
        except Exception as e:
            import sys
            sys.stderr.write(f"Firestore list_private_facts failed: {e}. Falling back to SQLite.\n")

    # SQLite Fallback
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

    # Remove from private database (Firestore or SQLite fallback)
    if _use_firebase and _firestore_db is not None:
        try:
            docs = _firestore_db.collection("private_facts").where("user_key", "==", k).stream()
            for doc in docs:
                fact_val = doc.to_dict().get("fact", "").lower()
                if needle in fact_val:
                    doc.reference.delete()
                    removed_private += 1
            return {"ok": True, "removed_public": removed_public, "removed_private": removed_private}
        except Exception as e:
            import sys
            sys.stderr.write(f"Firestore remove_fact failed: {e}. Falling back to SQLite.\n")

    # SQLite Fallback
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
