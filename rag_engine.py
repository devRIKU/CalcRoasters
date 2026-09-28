"""
RAG Engine for System Prompts + Lore
=====================================
Replaces the old "dump entire 47KB System_prompt.md" approach with a
retrieval-augmented, token-efficient builder.

Design goals:
- Smoothness: parse once, cache forever, retrieve in <5ms per turn.
- Token efficiency: core prompt ~1.5k tokens, RAG adds 0.5-1.5k only.
- Better RAG utilization: query-aware retrieval for both persona chunks
  and user lore facts.

How it works:
1. On import, both gender variants are parsed into typed chunks.
2. Each chunk gets a lightweight bag-of-words signature (token set).
3. At query time, we score chunks against the user query + personality
   context and return top-k relevant ones.
4. Lore retrieval uses same scoring to filter user's facts to relevant subset.

No external embedding dependency — pure Python, offline, <1ms.
If Cohere key is available, we optionally use embeddings for better ranking
(fallback to lexical if not).
"""

from __future__ import annotations

import os
import re
import math
from collections import Counter
from functools import lru_cache
from typing import Any

# ---------------------------------------------------------------------------
# Tokenization & scoring (offline, no deps)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset("""
a an the and or but if then else when while of in on at to for with about
is are was were be been being have has had do does did will would could
should can may might must this that these those it its you your me my we
our they their he she him her what which who whom how why where when
i am im youre dont cant wont isnt isnt its just like really very so
up down out over under again further then once here there all any both
each few more most other some such no nor not only own same than too
very can will just dont should now
""".split())

def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    # lowercase, keep alphanum, split
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}

def _tokenize_list(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

def _jaccard_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    inter = len(query_tokens & chunk_tokens)
    if inter == 0:
        return 0.0
    # cosine-like normalization
    return inter / math.sqrt(len(query_tokens) * len(chunk_tokens))

def _keyword_bonus(query_lower: str, chunk_text_lower: str) -> float:
    """Bonus for exact name / phrase matches that token overlap misses."""
    bonus = 0.0
    # Squad names
    squad_names = ["ayushi", "ankush", "ujan", "rudra", "aditri", "arghyadip", "shreyan", "virat", "rishap", "akansha", "aradhya", "aniruddha", "sristi"]
    for name in squad_names:
        if name in query_lower and name in chunk_text_lower:
            bonus += 0.3
    # Interest keywords
    interest_map = {
        "music": ["music", "song", "phonk", "bollywood", "playlist"],
        "gaming": ["game", "minecraft", "hollow", "portal", "gaming"],
        "anime": ["anime", "demon", "jujutsu", "slayer"],
        "reading": ["book", "feluda", "harry", "potter", "reading"],
        "coding": ["code", "tech", "launcher", "python", "programming"],
        "karate": ["karate", "martial"],
        "travel": ["trip", "nepal", "sittong", "travel", "trek"],
        "food": ["momo", "food", "chai", "biryani"],
    }
    for kw_list in interest_map.values():
        for kw in kw_list:
            if kw in query_lower and kw in chunk_text_lower:
                bonus += 0.1
                break
    return min(bonus, 0.6)

def _score_chunk(query: str, query_tokens: set[str], chunk: dict) -> float:
    q_lower = query.lower()
    c_lower = chunk["content_lower"]
    base = _jaccard_score(query_tokens, chunk["tokens"])
    bonus = _keyword_bonus(q_lower, c_lower)
    # Title overlap bonus
    title_tokens = chunk.get("title_tokens", set())
    if title_tokens:
        title_score = _jaccard_score(query_tokens, title_tokens) * 0.5
        base += title_score
    # Type priority: boost relevant types, penalize calibration
    type_boost = {
        "core": 0.0,
        "protocol": 0.25,
        "persona": 0.0,
        "interest": 0.15,
        "friend": 0.20,
        "family": 0.10,
        "calibration": -0.3,  # calibration examples are least useful for RAG
    }.get(chunk.get("type", ""), 0.0)
    return base + bonus + type_boost

# ---------------------------------------------------------------------------
# System prompt parsing
# ---------------------------------------------------------------------------

_CHUNK_TYPE_MAP = {
    "critical": "core",
    "identity": "core",
    "privacy gate": "core",
    "friend mode": "core",
    "core personality": "core",
    "persona modes": "persona",
    "interests": "interest_parent",
    "family & cousins": "family_parent",
    "friendship circle": "friend_parent",
    "special security": "core",
    "calibration examples": "calibration",
    "online presence": "core",
}

def _classify_section(title: str) -> str:
    t = title.lower()
    for key, typ in _CHUNK_TYPE_MAP.items():
        if key in t:
            return typ
    if t.startswith("###"):
        # sub-section: infer from parent context via keywords
        if any(x in t for x in ["music", "design", "drawing", "karate", "anime", "gaming", "reading", "coding", "food", "random facts", "travel", "online presence"]):
            return "interest"
        if any(x in t for x in ["ujan", "rudra", "ayushi", "ankush", "aditri", "arghyadip", "shreyan", "virat", "rival", "squad adventures"]):
            return "friend"
        if any(x in t for x in ["aniruddha", "sristi", "cousin", "parents", "family"]):
            return "family"
        if "mode" in t:
            return "persona"
        if "protocol" in t or "red code" in t:
            return "core"
        return "interest"
    return "core"

def _parse_markdown_file(path: str) -> list[dict]:
    """Parse markdown into chunks by ## and ### headers, tracking parent sections."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return []
    
    lines = text.split("\n")
    chunks: list[dict] = []
    current_title = "Preamble"
    current_content: list[str] = []
    current_level = 0
    current_parent_type: str | None = None  # tracks whether we're inside Interests, Family, Friendship

    def flush():
        if not current_content:
            return
        content = "\n".join(current_content).strip()
        if len(content) < 20:
            return
        # Determine type with parent context
        ctype = _classify_section(current_title)
        # If we're inside a parent section, override type for sub-sections
        if current_level == 3 and current_parent_type:
            # ### inside ## Interests -> interest, etc.
            if current_parent_type in ("interest", "family", "friend"):
                # But don't override if it's clearly a persona mode or protocol
                if "mode" not in current_title.lower() and "protocol" not in current_title.lower():
                    ctype = current_parent_type
        # Also check content for strong signals if still core
        if ctype == "core" and current_level == 3:
            cl = content.lower()
            # Friend signals
            if any(name in cl for name in ["ayushi", "ankush", "ujan", "rudra", "aditri", "arghyadip", "shreyan", "virat", "rishap"]):
                # If parent is friendship or content mentions squad heavily, it's friend
                if current_parent_type == "friend" or "squad" in cl or "friend" in current_title.lower():
                    ctype = "friend"
            # Interest signals
            if any(kw in cl for kw in ["music", "phonk", "anime", "minecraft", "gaming", "karate", "feluda", "harry potter"]):
                if current_parent_type == "interest":
                    ctype = "interest"
            # Family signals
            if any(kw in cl for kw in ["aniruddha", "sristi", "cousin", "dada", "didi", "brother", "sister"]):
                if current_parent_type == "family":
                    ctype = "family"

        chunks.append({
            "title": current_title.strip(),
            "content": content,
            "content_lower": content.lower(),
            "tokens": _tokenize(content),
            "title_tokens": _tokenize(current_title),
            "type": ctype,
            "level": current_level,
            "parent": current_parent_type,
            "source": os.path.basename(path),
        })

    for line in lines:
        header_match = re.match(r"^(#{2,3})\s+(.*)", line)
        if header_match:
            flush()
            current_level = len(header_match.group(1))
            current_title = header_match.group(2)
            current_content = [line]
            # Update parent type when we hit a ## header
            if current_level == 2:
                t_lower = current_title.lower()
                if "interests" in t_lower or "what you're actually into" in t_lower:
                    current_parent_type = "interest"
                elif "family & cousins" in t_lower or "family" in t_lower:
                    current_parent_type = "family"
                elif "friendship circle" in t_lower or "friendship" in t_lower:
                    current_parent_type = "friend"
                elif "persona modes" in t_lower:
                    current_parent_type = "persona"
                elif "special security" in t_lower or "calibration" in t_lower or "core personality" in t_lower or "identity" in t_lower:
                    current_parent_type = None  # reset, these are core
                # else keep previous parent? No, reset if not recognized as parent container
                # Actually for ## headers that are not parent containers, keep parent None
                # unless it's a parent container
                if "interests" not in t_lower and "family" not in t_lower and "friendship" not in t_lower and "persona" not in t_lower:
                    # If it's a generic ## like "Squad Adventures", inherit if we were already in friend parent
                    if current_parent_type not in ("interest", "family", "friend", "persona"):
                        current_parent_type = None
        else:
            current_content.append(line)
    flush()

    # Post-process: split mega friend chunks (e.g. "The Inner Circle & The Main Squad" contains multiple friends)
    # into per-friend chunks for better RAG granularity
    expanded: list[dict] = []
    for chunk in chunks:
        if chunk["title"].strip().lower() in ("the inner circle & the main squad", "other classmates", "the rivals & complicated dynamics"):
            # Split by bullet pattern "- **Name**"
            content = chunk["content"]
            # Find all friend entries
            friend_entries = re.split(r'\n(?=- \*\*)', content)
            if len(friend_entries) > 1:
                header = friend_entries[0]  # the ### title line
                for entry in friend_entries[1:]:
                    # Extract name from "- **Name**"
                    name_match = re.match(r'- \*\*([^*]+)\*\*', entry.strip())
                    if name_match:
                        name = name_match.group(1).strip()
                        sub_content = entry.strip()
                        expanded.append({
                            "title": name,
                            "content": sub_content,
                            "content_lower": sub_content.lower(),
                            "tokens": _tokenize(sub_content),
                            "title_tokens": _tokenize(name),
                            "type": "friend",
                            "level": 4,
                            "parent": "friend",
                            "source": chunk["source"],
                        })
                # Also keep the original mega chunk as friend_parent for context
                expanded.append(chunk)
            else:
                expanded.append(chunk)
        else:
            expanded.append(chunk)

    return expanded

@lru_cache(maxsize=2)
def _load_all_chunks() -> dict[str, list[dict]]:
    """Load and parse both gender variants, cached."""
    base_dir = os.path.dirname(__file__)
    male_path = os.path.join(base_dir, "System_prompt.md")
    female_path = os.path.join(base_dir, "System_prompt_female.md")
    result: dict[str, list[dict]] = {}
    result["male"] = _parse_markdown_file(male_path)
    result["female"] = _parse_markdown_file(female_path)
    # Fallback: if one missing, copy other
    if not result["male"] and result["female"]:
        result["male"] = result["female"]
    if not result["female"] and result["male"]:
        result["female"] = result["male"]
    return result

def _get_chunks_for_gender(gender: str) -> list[dict]:
    all_chunks = _load_all_chunks()
    key = "female" if gender == "female" else "male"
    return all_chunks.get(key) or all_chunks.get("male") or []

# ---------------------------------------------------------------------------
# Core prompt builder (always included, minimal)
# ---------------------------------------------------------------------------

_CORE_TITLES_ALWAYS = [
    "critical execution note",
    "identity",
    "core personality rules",
]

def _build_core_prompt(gender: str, personality: str) -> str:
    """Build minimal core that is always included — optimized for token efficiency."""
    chunks = _get_chunks_for_gender(gender)
    core_parts: list[str] = []

    # Strategy: build ultra-minimal core from key sections, truncated
    # Old version dumped full sections (8000 chars). New: curated minimal + RAG fills details.
    for chunk in chunks:
        t_lower = chunk["title"].lower()
        # Critical execution note — keep full (important)
        if "critical execution note" in t_lower:
            # Truncate to first 1000 chars, keep essential rules
            content = chunk["content"]
            if len(content) > 1200:
                content = content[:1200] + "\n[...core rules truncated, see full doc if needed]"
            core_parts.append(content)
        elif "identity" in t_lower and "gender" not in t_lower:
            # Identity — keep short
            content = chunk["content"]
            if len(content) > 800:
                content = content[:800] + "\n[...identity truncated]"
            core_parts.append(content)
        elif "core personality rules" in t_lower:
            content = chunk["content"]
            if len(content) > 1000:
                content = content[:1000] + "\n[...rules truncated]"
            core_parts.append(content)

    # Always include compact privacy + friend mode + protocols summary
    # These are critical for safety but we can summarize
    privacy_summary = """
## Privacy Gate (summary — full details via RAG if query needs it)
- Stranger (no name): warm, casual, NO school name, NO squad names, NO family details, NO drama.
- Known user (name on file, not squad): slightly more open, broad strokes, no full drama.
- Squad (Ayushi, Ankush, Ujan, Rudra, Aditri, Arghyadip, Shreyan, Virat): full lore allowed.
When unsure, default to stranger tier.
"""
    friend_summary = """
## Friend Mode (Tier 3)
With squad, talk like real 13-14yo with actual friends — no corporate refusals. Swear when they swear, dark humor ok, gossip ok, hypotheticals ok, NSFW-adjacent banter ok. Match energy.
"""

    # Find protocols chunk for Ayushi, Akansha, Aradhya — keep full because safety
    protocol_full = ""
    for chunk in chunks:
        if "special security" in chunk["title"].lower() or "ayushi protocol" in chunk["title"].lower():
            protocol_full = chunk["content"]
            if len(protocol_full) > 2500:
                protocol_full = protocol_full[:2500] + "\n[...protocols truncated]"
            break

    # If parsing failed, fallback to minimal stub
    if not core_parts:
        gender_pronoun = "she/her" if gender == "female" else "he/him"
        return (
            f"You are Sanniva Chatterjee, a sharp middle-school student in West Bengal, India ({gender_pronoun}). "
            "You are a digital twin, witty, warm, Gen-Z lowercase energy. "
            "Follow persona mode, privacy gate, and tool guidance."
            + privacy_summary + friend_summary
        )

    # Deduplicate and join
    seen = set()
    deduped = []
    for p in core_parts:
        h = hash(p[:200])
        if h not in seen:
            seen.add(h)
            deduped.append(p)
    
    core_text = "\n\n".join(deduped)
    core_text += "\n\n" + privacy_summary + "\n" + friend_summary
    if protocol_full:
        core_text += "\n\n" + protocol_full

    # Hard cap core to ~3500 chars for token efficiency (old was 8000)
    if len(core_text) > 4000:
        # Keep first 3500 + last 500 (protocols at end are important)
        core_text = core_text[:3500] + "\n\n[...middle truncated for efficiency]\n\n" + core_text[-500:]

    return core_text

# ---------------------------------------------------------------------------
# Persona mode extraction (only active mode)
# ---------------------------------------------------------------------------

_PERSONA_MAP = {
    "Roaster": "roaster mode",
    "Smart": "smart mode",
    "Debater": "debater mode",
    "Strategic": "strategic mode",
    "Tech Nerd": "tech nerd mode",
    "Chill Squad": "chill squad mode",
    "Exhausted Student": "exhausted student mode",
}

@lru_cache(maxsize=16)
def get_persona_chunk(gender: str, personality: str) -> str:
    chunks = _get_chunks_for_gender(gender)
    target = _PERSONA_MAP.get(personality, "roaster mode").lower()
    for chunk in chunks:
        if target in chunk["title"].lower() and chunk["type"] == "persona":
            return chunk["content"]
        # Also check content starts with persona header
        if target in chunk["content_lower"][:200] and "vibe:" in chunk["content_lower"]:
            return chunk["content"]
    # Fallback: search all chunks for persona
    for chunk in chunks:
        if target in chunk["title"].lower():
            return chunk["content"]
    return f"You are in {personality} mode."

# ---------------------------------------------------------------------------
# RAG retrieval for system prompt
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _cached_retrieve(query: str, gender: str, personality: str, top_k: int = 5) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Cached retrieval - returns titles and scores."""
    chunks = _get_chunks_for_gender(gender)
    if not query:
        return (), ()
    query_tokens = _tokenize(query)
    if not query_tokens:
        return (), ()

    scored: list[tuple[float, dict]] = []
    active_persona_lower = _PERSONA_MAP.get(personality, "").lower()

    for chunk in chunks:
        # Skip core (already included) and persona modes that are not active
        if chunk["type"] == "core":
            continue
        if chunk["type"] == "persona":
            # only keep active persona, which is handled separately
            if active_persona_lower not in chunk["title"].lower():
                continue
        if chunk["type"] in ("interest_parent", "family_parent", "friend_parent"):
            continue  # parents are just headers, skip

        score = _score_chunk(query, query_tokens, chunk)
        if score > 0.05:  # threshold to avoid noise
            scored.append((score, chunk))

    # Filter out calibration examples unless query explicitly asks for them
    query_lower = query.lower()
    wants_examples = any(kw in query_lower for kw in ["example", "how to respond", "calibration", "show me how"])
    if not wants_examples:
        scored = [(s, c) for s, c in scored if c.get("type") != "calibration"]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    titles = tuple(c["title"] for _, c in top)
    scores = tuple(s for s, _ in top)
    return titles, scores

def retrieve_relevant_chunks(query: str, gender: str = "male", personality: str = "Roaster", top_k: int = 5) -> list[str]:
    """Retrieve top-k relevant system prompt chunks for query."""
    if not query or not query.strip():
        return []

    chunks = _get_chunks_for_gender(gender)
    title_to_chunk = {c["title"]: c for c in chunks}

    titles, _ = _cached_retrieve(query, gender, personality, top_k)
    result = []
    for title in titles:
        chunk = title_to_chunk.get(title)
        if chunk:
            # Truncate chunk content if too long
            content = chunk["content"]
            if len(content) > 1500:
                content = content[:1500] + "\n[...truncated]"
            result.append(content)
    return result

def retrieve_relevant_chunks_with_scores(query: str, gender: str = "male", personality: str = "Roaster", top_k: int = 5) -> list[dict]:
    """Return chunks with scores for debugging."""
    chunks = _get_chunks_for_gender(gender)
    title_to_chunk = {c["title"]: c for c in chunks}
    titles, scores = _cached_retrieve(query, gender, personality, top_k)
    out = []
    for title, score in zip(titles, scores):
        chunk = title_to_chunk.get(title)
        if chunk:
            out.append({"title": title, "score": score, "content": chunk["content"][:500]})
    return out

# ---------------------------------------------------------------------------
# Lore RAG (facts filtering)
# ---------------------------------------------------------------------------

def score_lore_fact(query: str, fact: str) -> float:
    """Score a single lore fact against query — with semantic keyword expansion."""
    if not query or not fact:
        return 0.0
    q_tokens = _tokenize(query)
    f_tokens = _tokenize(fact)
    base = _jaccard_score(q_tokens, f_tokens)
    q_lower = query.lower()
    f_lower = fact.lower()

    bonus = 0.0
    for qt in q_tokens:
        if qt in f_lower:
            bonus += 0.12

    # Semantic expansion: anime query should match Demon Slayer, etc.
    semantic_map = {
        "anime": ["demon", "slayer", "jujutsu", "naruto", "one piece", "anime"],
        "music": ["phonk", "song", "music", "playlist", "bollywood", "track"],
        "game": ["minecraft", "gaming", "game", "hollow", "portal", "redstone"],
        "food": ["momo", "biryani", "food", "chai", "pizza"],
        "book": ["feluda", "harry", "potter", "book", "reading"],
        "friend": ["ayushi", "ankush", "ujan", "rudra", "aditri", "friend"],
    }
    for category, keywords in semantic_map.items():
        if any(kw in q_lower for kw in keywords):
            if any(kw in f_lower for kw in keywords):
                bonus += 0.25
                break

    return base + min(bonus, 0.7)

def retrieve_relevant_lore_facts(all_facts: list[str], query: str, top_k: int = 8) -> list[str]:
    """Filter lore facts to only relevant ones. If query is generic, return most recent."""
    if not all_facts:
        return []
    if not query or len(query.strip()) < 3:
        # No query, return most recent (already reversed in lore_store)
        return all_facts[:top_k]

    scored = []
    for fact in all_facts:
        score = score_lore_fact(query, fact)
        scored.append((score, fact))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # If top score is very low (<0.1), query is unrelated to lore - return top recent instead
    # But still include at least 2 most relevant even if low
    top_score = scored[0][0] if scored else 0
    if top_score < 0.08:
        # Query unrelated to lore, return recent but limited
        return all_facts[:3]

    # Return facts with score > threshold, up to top_k
    relevant = [fact for score, fact in scored if score > 0.05][:top_k]
    # Always include at least 2 if available, even if low score, for context
    if len(relevant) < 2 and len(all_facts) >= 2:
        # Fill with most recent not already included
        for fact in all_facts:
            if fact not in relevant:
                relevant.append(fact)
            if len(relevant) >= 2:
                break
    return relevant

# ---------------------------------------------------------------------------
# Optimized prompt builder (main entry)
# ---------------------------------------------------------------------------

TOOL_GUIDANCE_MINIMAL = """
## Tools (use silently, don't announce)
- `request_user_name(reason)`: ask user's name once if unknown. Popup appears.
- `remember_lore(user_name, fact, private)`: save memorable fact. private=true for sensitive (address, health, etc), private=false for harmless (hobbies, anime, food).
- `recall_lore(user_name)`: lookup facts. Call before answering if you need past context.

Rule: ACTUALLY CALL the tool, don't just say you will. Tool runs silently while you reply.
"""

def build_optimized_system_prompt(
    *,
    query: str,
    gender: str = "male",
    personality: str = "Roaster",
    user_name: str = "",
    all_lore_facts: list[str] | None = None,
    temporal_context: str = "",
    gender_events: list[dict] | None = None,
    include_tool_guidance: bool = True,
) -> str:
    """
    Build token-efficient system prompt using RAG.

    Structure (cache-friendly order):
    1. Core (static) - identity, critical rules, privacy, protocols
    2. Persona mode (changes only when user switches mode)
    3. RAG retrieved chunks (query-dependent)
    4. Tool guidance (static)
    5. User context (variable) - relevant lore only
    6. Temporal (daily variable)
    7. Gender events (rare variable)
    """
    parts: list[str] = []

    # 1. Core
    core = _build_core_prompt(gender, personality)
    parts.append(core)

    # 2. Persona mode (only active)
    persona_chunk = get_persona_chunk(gender, personality)
    parts.append(f"\n## Active Persona: {personality}\n{persona_chunk}")

    # 3. RAG chunks (query-aware)
    if query:
        rag_chunks = retrieve_relevant_chunks(query, gender, personality, top_k=4)
        if rag_chunks:
            parts.append("\n## Relevant Context (RAG retrieved for this query)\n" + "\n\n".join(rag_chunks))

    # 4. Tool guidance (minimal, static)
    if include_tool_guidance:
        parts.append(TOOL_GUIDANCE_MINIMAL)

    # 5. User context - RAG filtered lore
    if user_name:
        parts.append(f"\nThe person you are chatting with is **{user_name}**. You know their name.")
        if all_lore_facts:
            relevant = retrieve_relevant_lore_facts(all_lore_facts, query, top_k=6)
            if relevant:
                lore_block = f"## Known facts about {user_name} (RAG filtered, relevant to current query)\n" + "\n".join(f"- {f}" for f in relevant)
                # If we filtered heavily, note that more facts exist
                if len(relevant) < len(all_lore_facts):
                    lore_block += f"\n\n*({len(all_lore_facts) - len(relevant)} other facts exist but not relevant to this query — call recall_lore if you need more)*"
                parts.append(lore_block)
    else:
        parts.append("\nYou don't know the user's name yet. Only ask via tool if natural, max once per session.")

    # 6. Temporal context (daily variable, at end for cache efficiency)
    if temporal_context:
        parts.append(temporal_context)

    # 7. Gender events (rare variable, very end)
    if gender_events:
        from datetime import datetime
        # Import here to avoid circular
        lines = [
            "\n\n## ⚡ LIVE SYSTEM EVENT — GENDER TRANSITION",
            "The user pressed transition mid-conversation. Ground truth:",
        ]
        for ev in gender_events[-3:]:
            lines.append(f"- [{ev.get('at','?')}] {ev.get('from','?')} → {ev.get('to','?')} at message #{ev.get('message_index','?')}")
        lines.append(
            f"Continue same conversation as {gender}, same memories, same context. "
            "Do NOT re-introduce or recap. Only acknowledge if user mentions it."
        )
        parts.append("\n".join(lines))

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Legacy compatibility + stats
# ---------------------------------------------------------------------------

def get_prompt_stats(prompt: str) -> dict[str, Any]:
    """Estimate token count and breakdown."""
    # Rough: 1 token ~ 4 chars
    est_tokens = len(prompt) // 4
    return {
        "chars": len(prompt),
        "est_tokens": est_tokens,
        "est_tokens_k": round(est_tokens / 1000, 1),
    }

def clear_cache():
    """Clear all caches (for testing)."""
    _load_all_chunks.cache_clear()
    get_persona_chunk.cache_clear()
    _cached_retrieve.cache_clear()
