"""Free unlimited local web search for InterviewCopilot.

No API key, no browser/chromium. Uses httpx + DuckDuckGo HTML endpoint
(https://html.duckduckgo.com/html/) and stdlib regex/html parsing.

Never raises: failures return a "Search failed: ..." string.
Public API is exported via ``__all__``.
"""

from __future__ import annotations

import asyncio
import html as _html
import math
import re
from typing import Any

__all__ = [
    "search",
    "search_sync",
    "web_search_tool",
]

_DDG_URL: str = "https://html.duckduckgo.com/html/"
_UA: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

_LINK_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:div|td)',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_REDIRECT_RE = re.compile(r"uddg=([^&]+)")


def _clamp_max_results(value: Any, default: int = 5) -> int:
    """Coerce *value* to an int result count in [1, 10]."""
    if isinstance(value, bool):
        return default
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(10, n))


def _clamp_timeout(value: Any, default: float = 12.0) -> float:
    """Coerce *value* to a finite timeout in [1.0, 30.0] seconds."""
    if isinstance(value, bool):
        return default
    try:
        t = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(t):
        return default
    return max(1.0, min(30.0, t))


def _clean(raw: str) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    try:
        text = _TAG_RE.sub("", raw or "")
        text = _html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def _parse_results(page: str, max_results: int) -> list[tuple[str, str, str]]:
    """Parse DuckDuckGo HTML page into ``(title, url, snippet)`` tuples."""
    n = _clamp_max_results(max_results)
    if not isinstance(page, str):
        return []
    out: list[tuple[str, str, str]] = []
    try:
        links = _LINK_RE.findall(page or "")
        snippets = _SNIPPET_RE.findall(page or "")
    except Exception:
        return out
    for i, (href, title_html) in enumerate(links[:n]):
        try:
            title = _clean(title_html) or "(no title)"
            url = _html.unescape(href.strip())
            # unwrap DuckDuckGo redirect links //duckduckgo.com/l/?uddg=<target>
            m = _REDIRECT_RE.search(url)
            if m:
                try:
                    from urllib.parse import unquote

                    url = unquote(m.group(1))
                except Exception:
                    pass
            snippet = _clean(snippets[i]) if i < len(snippets) else ""
            out.append((title, url, snippet))
        except Exception:
            continue
    return out


def search_sync(query: str, max_results: int = 5, timeout: float = 12.0) -> str:
    """Blocking DuckDuckGo html search. Returns a formatted results string.

    Empty queries return ``"Search failed: empty query"``; a page with no
    parseable results returns ``"No results found for: <query>"``. Never
    raises — failures return ``"Search failed: ..."``.
    """
    try:
        q = str(query or "").strip()
        if not q:
            return "Search failed: empty query"
        n = _clamp_max_results(max_results)
        t = _clamp_timeout(timeout)
        try:
            import httpx
        except Exception as exc:
            return f"Search failed: httpx not installed ({exc})"
        try:
            resp = httpx.post(
                _DDG_URL,
                data={"q": q},
                headers={"User-Agent": _UA},
                timeout=t,
                follow_redirects=True,
            )
            resp.raise_for_status()
            page = resp.text
        except Exception as exc:
            return f"Search failed: {exc}"
        results = _parse_results(page, n)
        if not results:
            return f"No results found for: {q}"
        lines: list[str] = []
        for i, (title, url, snippet) in enumerate(results, 1):
            lines.append(f"{i}. {title}\n{url}\n{snippet}".rstrip())
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Search failed: {exc}"


async def search(query: str, max_results: int = 5, timeout: float = 12.0) -> str:
    """Async wrapper around :func:`search_sync` (runs in a worker thread)."""
    try:
        return await asyncio.to_thread(search_sync, query, max_results, timeout)
    except Exception as exc:
        return f"Search failed: {exc}"


def web_search_tool(query: str, max_results: int = 5, timeout: float = 12.0) -> str:
    """Sync wrapper for Live function-calling. Never raises."""
    try:
        return search_sync(query, max_results, timeout)
    except Exception as exc:
        return f"Search failed: {exc}"
