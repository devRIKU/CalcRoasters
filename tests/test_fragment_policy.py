"""Static guard: fragments must never own sidebar (or other out-of-container) widgets.

Why this exists
---------------
`chatbot.py` renders most of its UI in the sidebar. An earlier "optimization"
wrapped two sidebar builders in `@st.fragment` so that tuning a slider wouldn't
redraw the chat transcript:

    @_fragment
    def _sidebar_model_settings() -> float: ...        # st.sidebar.button/slider
    @_fragment
    def _sidebar_model_chain_picker() -> None: ...     # -> helpers -> st.sidebar.*

Streamlit forbids that: a widget created by a fragment must live inside the
fragment's own container. The sidebar is a separate root container, so its
delta path (`[1, x, ...]`) is never prefixed by the fragment's path
(`[0, k]`) and `streamlit/elements/lib/policies.py::check_fragment_path_policy`
raises:

    StreamlitFragmentWidgetsNotAllowedOutsideError:
        Fragments cannot write widgets to outside containers.

Because the check runs on every execution of the fragment (full app run
included), the app died on page load — the first widget in the block,
`st.sidebar.button("🗑️ Clear Chat")`, raised before the chat UI rendered.

The trickiest part of that bug is *indirect* rendering: `_sidebar_model_chain_picker`
never called `st.sidebar.multiselect` itself, it delegated to
`_sidebar_provider_picker` / `_sidebar_manual_provider_override`. So this test
walks the call graph of every fragment, not just its direct body.

This test parses the source instead of importing the app so it needs no
Streamlit install, no API keys and no network: it fails fast in CI or on a
dev box the moment somebody re-adds an illegal fragment.

Run:  python tests/test_fragment_policy.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Element calls that do NOT create widgets. The fragment path policy only
# polices widgets, so a fragment *may* technically paint these into a foreign
# container (that is what `_render_tool_status_banner` relies on for captions).
# Everything else — including layout containers, which would put their children
# outside the fragment too — is treated as a violation.
SAFE_ELEMENT_CALLS = frozenset(
    {
        "audio",
        "caption",
        "code",
        "dataframe",
        "divider",
        "error",
        "exception",
        "header",
        "help",
        "html",
        "image",
        "info",
        "json",
        "latex",
        "markdown",
        "metric",
        "pyplot",
        "subheader",
        "success",
        "table",
        "text",
        "title",
        "toast",
        "video",
        "warning",
        "write",
    }
)

# Layout calls that are elements themselves but *create a container* in the
# sidebar root. The container call alone is tolerated by Streamlit, yet every
# widget placed inside it (or in the `with` block) lives outside the fragment
# and raises — so treat any of these as a violation too.
SIDEBAR_CONTAINER_CALLS = frozenset(
    {
        "columns",
        "container",
        "dialog",
        "empty",
        "expander",
        "form",
        "popover",
        "status",
        "tabs",
    }
)

# Sidebar calls that touch no container at all, so they are harmless inside a
# fragment (plain elements, alert banners).
SIDEBAR_SAFE_ATTRS = SAFE_ELEMENT_CALLS

# Fragments that are known-good and must stay decoratable (main-body renders).
EXPECTED_FRAGMENTS = {
    "_render_tool_status_banner",
    "_flush_lore_confirmations",
    "_maybe_show_name_popup",
}

# Functions that render sidebar widgets and must never be fragment-decorated.
MUST_NOT_BE_FRAGMENTS = {
    "_sidebar_model_settings",
    "_sidebar_model_chain_picker",
}


def _decorator_name(node: ast.expr) -> str:
    """Best-effort textual name of a decorator expression."""
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - very old Pythons
        return getattr(node, "id", "")


def _is_fragment_decorator(node: ast.expr) -> bool:
    return "fragment" in _decorator_name(node).lower()


def _is_fragment_fn(fn: ast.FunctionDef) -> bool:
    return any(_is_fragment_decorator(d) for d in fn.decorator_list)


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Module-level `def`s (the only ones a fragment can call by bare name)."""
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _called_names(fn: ast.AST) -> set[str]:
    """Bare function names called anywhere inside `fn` (nested defs included)."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                names.add(func.value.id)
    return names


def _sidebar_widget_calls(fn: ast.AST) -> list[tuple[int, str, str]]:
    """`st.sidebar.<call>()` (and alias) hits inside `fn` as (line, attr, text)."""
    hits: list[tuple[int, str, str]] = []
    aliases: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if ast.unparse(node.value) == "st.sidebar":
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr in SIDEBAR_SAFE_ATTRS:
            continue
        base = node.func.value
        if isinstance(base, ast.Attribute) and ast.unparse(base) == "st.sidebar":
            hits.append((node.lineno, attr, f"st.sidebar.{attr}()"))
        elif isinstance(base, ast.Name) and base.id in aliases:
            hits.append(
                (node.lineno, attr, f"{base.id}.{attr}()  # alias of st.sidebar")
            )
    return hits


def _reachable(fragment: ast.FunctionDef, index: dict[str, ast.FunctionDef]) -> list[ast.FunctionDef]:
    """`fragment` plus every module-level function it calls, transitively."""
    seen: list[ast.FunctionDef] = []
    queue: list[ast.FunctionDef] = [fragment]
    visited: set[str] = set()
    while queue:
        current = queue.pop()
        seen.append(current)
        for name in _called_names(current):
            if name in index and name not in visited and index[name] is not fragment:
                visited.add(name)
                queue.append(index[name])
    return seen


def find_violations(path: Path) -> list[str]:
    """Report every fragment that can reach a sidebar (or aliased) widget call."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    index = _module_functions(tree)
    problems: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not _is_fragment_fn(node):
            continue
        reachable = [
            (node.name, node),
            *((f.name, f) for f in _reachable(node, index) if f is not node),
        ]
        for fname, fn in reachable:
            for lineno, attr, call in _sidebar_widget_calls(fn):
                if fname == node.name:
                    chain = f"`{node.name}` is a fragment but calls `{call}`"
                else:
                    chain = (
                        f"`{node.name}` is a fragment and reaches `{fname}`, "
                        f"which calls `{call}`"
                    )
                if attr in SIDEBAR_CONTAINER_CALLS:
                    why = (
                        "widgets placed inside that container are outside the "
                        "fragment, so Streamlit raises "
                        "StreamlitFragmentWidgetsNotAllowedOutsideError"
                    )
                else:
                    why = (
                        "Streamlit raises "
                        "StreamlitFragmentWidgetsNotAllowedOutsideError"
                    )
                problems.append(f"{path.name}:{lineno}: {chain} — {why}.")
    return problems


def check_named_functions(path: Path) -> list[str]:
    """Assert the specific functions from the incident are not decorated."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorated = _is_fragment_fn(node)
        if node.name in MUST_NOT_BE_FRAGMENTS:
            seen.add(node.name)
            if decorated:
                problems.append(
                    f"{path.name}:{node.lineno}: `{node.name}` renders sidebar "
                    f"widgets and must not be wrapped in @st.fragment."
                )
        elif decorated and node.name in EXPECTED_FRAGMENTS:
            seen.add(node.name)

    for missing in sorted(set(MUST_NOT_BE_FRAGMENTS | EXPECTED_FRAGMENTS) - seen):
        problems.append(
            f"{path.name}: expected function `{missing}` was not found — update "
            f"tests/test_fragment_policy.py if it was renamed or removed."
        )
    return problems


def main() -> int:
    sources = [REPO_ROOT / "chatbot.py"] + sorted(REPO_ROOT.glob("*.py"))
    target = REPO_ROOT / "chatbot.py"

    problems: list[str] = []
    for path in dict.fromkeys(sources):  # dedupe, keep order
        if path.exists():
            problems.extend(find_violations(path))
    if target.exists():
        problems.extend(check_named_functions(target))

    if problems:
        print("Fragment policy violations found:\n")
        for p in problems:
            print(f"  ✗ {p}")
        print(
            "\nFragments may not create widgets outside their own container. "
            "Sidebar widgets must be rendered by the main script."
        )
        return 1

    print("OK — no fragment writes widgets outside its own container.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
