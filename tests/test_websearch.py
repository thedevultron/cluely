"""Tests for websearch: stdlib + pytest only. httpx is mocked; no network."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest import mock

import pytest

import websearch as ws


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

DDG_HTML = """
<html><body>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">Title One &amp; More</a>
<a class="result__snippet" href="#">Snippet for one &amp; more</a>
<a class="result__a" href="https://example.com/two">Second Result</a>
<a class="result__snippet" href="#">Second snippet text</a>
</body></html>
"""

# snippet regex requires class="result__snippet" ... </div> or </td> closing;
# ensure fixture uses </div> so _parse_results pairs correctly
DDG_HTML2 = """
<html><body>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">Title One &amp; More</a>
<div class="result__snippet">Snippet for one &amp; more</div>
<a class="result__a" href="https://example.com/two">Second Result</a>
<div class="result__snippet">Second snippet text</div>
</body></html>
"""


class FakeResp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_httpx_post(monkeypatch, resp_or_exc):
    """Inject a fake httpx module with our post()."""
    fake = types.ModuleType("httpx")

    def _post(*a, **kw):
        _post.calls.append((a, kw))
        if isinstance(resp_or_exc, BaseException):
            raise resp_or_exc
        return resp_or_exc

    _post.calls = []  # type: ignore[attr-defined]
    fake.post = _post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return _post


# ---------------------------------------------------------------------------
# _parse_results
# ---------------------------------------------------------------------------

def test_all_exports():
    assert hasattr(ws, "__all__")
    for name in ("search_sync", "search", "web_search_tool"):
        assert name in ws.__all__


def test_parse_results_basic():
    out = ws._parse_results(DDG_HTML2, 5)
    assert len(out) == 2
    assert out[0][0] == "Title One & More"  # html unescaped
    assert out[0][1] == "https://example.com/one"  # redirect unwrapped
    assert "Snippet for one" in out[0][2]
    assert out[1][0] == "Second Result"
    assert out[1][1] == "https://example.com/two"


def test_parse_results_empty_and_none():
    assert ws._parse_results("", 5) == []
    assert ws._parse_results(None, 5) == []  # type: ignore[arg-type]
    assert ws._parse_results("<html>no results</html>", 5) == []


def test_parse_results_max_results_clamp():
    out = ws._parse_results(DDG_HTML2, 1)
    assert len(out) == 1
    # bad max_results must not crash
    assert isinstance(ws._parse_results(DDG_HTML2, "bad"), list)  # type: ignore[arg-type]
    assert isinstance(ws._parse_results(DDG_HTML2, None), list)  # type: ignore[arg-type]
    assert isinstance(ws._parse_results(DDG_HTML2, -5), list)
    assert isinstance(ws._parse_results(DDG_HTML2, 9999), list)


def test_clean_strips_tags_and_entities():
    assert ws._clean("<b>Hello</b> &amp; welcome") == "Hello & welcome"
    assert ws._clean(None) == ""  # type: ignore[arg-type]
    assert ws._clean("") == ""


# ---------------------------------------------------------------------------
# search_sync
# ---------------------------------------------------------------------------

def test_search_sync_empty_query():
    assert "empty query" in ws.search_sync("")
    assert "empty query" in ws.search_sync("   ")
    assert "empty query" in ws.search_sync(None)  # type: ignore[arg-type]


def test_search_sync_success(monkeypatch):
    post = _patch_httpx_post(monkeypatch, FakeResp(DDG_HTML2))
    out = ws.search_sync("python asyncio", max_results=5)
    assert "Title One" in out
    assert "https://example.com/one" in out
    assert post.calls[0][1]["data"]["q"] == "python asyncio"
    assert post.calls[0][1]["headers"]["User-Agent"]


def test_search_sync_no_results(monkeypatch):
    _patch_httpx_post(monkeypatch, FakeResp("<html>empty</html>"))
    out = ws.search_sync("nothing matches this", 5)
    assert out.startswith("No results found")


def test_search_sync_network_failure(monkeypatch):
    _patch_httpx_post(monkeypatch, RuntimeError("boom"))
    out = ws.search_sync("query", 5)
    assert out.startswith("Search failed")
    assert "boom" in out


def test_search_sync_http_error(monkeypatch):
    _patch_httpx_post(monkeypatch, FakeResp("err", status=500))
    out = ws.search_sync("query", 5)
    assert out.startswith("Search failed")


def test_search_sync_bad_timeout_and_max_results(monkeypatch):
    post = _patch_httpx_post(monkeypatch, FakeResp(DDG_HTML2))
    # bad timeouts must not crash and must be sanitized to a sane float
    for bad_timeout in (None, "bad", -5, 0, float("inf"), float("nan"), True, 999):
        out = ws.search_sync("q", max_results=5, timeout=bad_timeout)  # type: ignore[arg-type]
        assert isinstance(out, str)
    # last call timeout should have been clamped to sane range
    last_timeout = post.calls[-1][1]["timeout"]
    assert 1.0 <= float(last_timeout) <= 30.0
    # bad max_results
    assert isinstance(ws.search_sync("q", max_results="bad"), str)  # type: ignore[arg-type]
    assert isinstance(ws.search_sync("q", max_results=None), str)  # type: ignore[arg-type]
    assert isinstance(ws.search_sync("q", max_results=-1), str)


def test_search_sync_httpx_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    # None in sys.modules makes `import httpx` raise ImportError
    out = ws.search_sync("query", 5)
    assert out.startswith("Search failed")


def test_search_async_wrapper(monkeypatch):
    _patch_httpx_post(monkeypatch, FakeResp(DDG_HTML2))
    out = asyncio.run(ws.search("async query", 2))
    assert "Title One" in out
    assert asyncio.run(ws.search("", 2)).startswith("Search failed")


def test_web_search_tool(monkeypatch):
    _patch_httpx_post(monkeypatch, FakeResp(DDG_HTML2))
    assert "Title One" in ws.web_search_tool("tool query")
    assert ws.web_search_tool("").startswith("Search failed")
    # extended signature: max_results/timeout kwargs (backward compat)
    out = ws.web_search_tool("tool query", max_results=1, timeout=5)
    assert "Title One" in out
