"""
Performance utilities — smoothness optimizations
=================================================
Centralizes performance improvements:
- Timing decorators
- Cache warming
- Memory optimization
- Streamlit fragment helpers
"""

from __future__ import annotations

import time
import functools
from typing import Any, Callable

import streamlit as st

def timed(name: str = ""):
    """Decorator to measure function execution time and log to session state."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.time() - start) * 1000
                try:
                    perf_log = st.session_state.setdefault("_perf_log", [])
                    perf_log.append({"name": name or fn.__name__, "ms": round(elapsed, 1), "ts": time.time()})
                    # Keep last 20
                    st.session_state["_perf_log"] = perf_log[-20:]
                except Exception:
                    pass
        return wrapper
    return decorator

@st.cache_data(ttl=3600, show_spinner=False)
def get_system_info() -> dict[str, Any]:
    """Cached system info for debugging."""
    import os, sys
    return {
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }

def warmup_caches():
    """Pre-warm all caches to avoid first-query latency."""
    try:
        import rag_engine
        rag_engine._load_all_chunks()
    except Exception:
        pass
    try:
        import lore_store
        # Warm lore cache by loading empty
        lore_store._load()
    except Exception:
        pass

def clear_all_caches():
    """Clear all Streamlit and RAG caches."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    try:
        import rag_engine
        rag_engine.clear_cache()
    except Exception:
        pass
    try:
        from styles import clear_theme_cache
        clear_theme_cache()
    except Exception:
        pass

def get_perf_summary() -> str:
    """Get performance log summary for sidebar."""
    try:
        log = st.session_state.get("_perf_log", [])
        if not log:
            return "No perf data yet"
        # Group by name, avg
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)
        for entry in log:
            groups[entry["name"]].append(entry["ms"])
        lines = []
        for name, times in groups.items():
            avg = sum(times) / len(times)
            lines.append(f"{name}: {avg:.1f}ms avg ({len(times)} samples)")
        return "\n".join(lines)
    except Exception:
        return "Perf log unavailable"
