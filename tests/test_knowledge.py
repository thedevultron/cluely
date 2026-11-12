"""Tests for knowledge: stdlib + pytest only, no network."""

from __future__ import annotations

import pytest

import knowledge as kb


def test_all_exports():
    assert hasattr(kb, "__all__")
    for name in ("list_knowledge_files", "load_knowledge"):
        assert name in kb.__all__
    assert "TEXT_EXTENSIONS" in kb.__all__
    assert "MAX_FILE_BYTES" in kb.__all__


def _write(path, text: str, encoding: str = "utf-8") -> str:
    path.write_text(text, encoding=encoding)
    return str(path)


def test_list_nonexistent_returns_empty():
    assert kb.list_knowledge_files([]) == []
    assert kb.list_knowledge_files(None) == []  # type: ignore[arg-type]
    assert kb.list_knowledge_files(["/does/not/exist.txt"]) == []
    assert kb.list_knowledge_files(["", "  "]) == []


def test_list_string_path_not_split_into_chars(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "hello")
    # passing a bare string (not a list) must be treated as one path
    found = kb.list_knowledge_files(str(f))  # type: ignore[arg-type]
    assert len(found) == 1


def test_list_file_and_dir_walk(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "sub" / "b.md"
    c = tmp_path / "sub" / "ignore.exe"
    d = tmp_path / "sub" / "notes.rst"
    b.parent.mkdir(parents=True)
    _write(a, "aaa")
    _write(b, "bbb")
    _write(c, "not really binary but extension excluded from dir walk")
    _write(d, "ddd")
    found = kb.list_knowledge_files([str(tmp_path)])
    names = {f.split("\\")[-1].split("/")[-1] for f in found}
    assert "a.txt" in names
    assert "b.md" in names
    assert "notes.rst" in names
    assert "ignore.exe" not in names  # dir walk only allows TEXT_EXTENSIONS


def test_list_explicit_binary_skipped(tmp_path):
    png = tmp_path / "img.png"
    txt = tmp_path / "ok.txt"
    _write(png, "not really a png")
    _write(txt, "ok")
    found = kb.list_knowledge_files([str(png), str(txt)])
    assert len(found) == 1
    assert found[0].endswith("ok.txt")


def test_list_extensionless_allowed(tmp_path):
    f = tmp_path / "NOTES"
    _write(f, "plain text no extension")
    found = kb.list_knowledge_files([str(f)])
    assert len(found) == 1


def test_list_dedup(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "x")
    found = kb.list_knowledge_files([str(f), str(f), str(tmp_path)])
    assert len(found) == len(set(found))
    assert len(found) == 1


def test_load_knowledge_empty_inputs():
    assert kb.load_knowledge([], 12000) == ""
    assert kb.load_knowledge(None, 12000) == ""  # type: ignore[arg-type]
    assert kb.load_knowledge(["/does/not/exist.txt"], 12000) == ""


def test_load_knowledge_concatenates_with_header(tmp_path):
    f1 = tmp_path / "one.txt"
    f2 = tmp_path / "two.md"
    _write(f1, "hello one")
    _write(f2, "hello two")
    out = kb.load_knowledge([str(f1), str(f2)], 12000)
    assert "===== FILE: one.txt =====" in out
    assert "===== FILE: two.md =====" in out
    assert "hello one" in out
    assert "hello two" in out


def test_load_knowledge_truncated_never_exceeds_budget(tmp_path):
    f = tmp_path / "big.txt"
    _write(f, "x" * 50000)
    for budget in (1000, 2000, 5000, 12000):
        out = kb.load_knowledge([str(f)], budget)
        assert len(out) <= budget, f"budget {budget} exceeded: {len(out)}"
        assert len(out) > 0


def test_load_knowledge_multi_file_budget(tmp_path):
    for i in range(5):
        _write(tmp_path / f"f{i}.txt", f"content-{i}-" + "y" * 4000)
    out = kb.load_knowledge([str(tmp_path / f"f{i}.txt") for i in range(5)], 3000)
    assert len(out) <= 3000
    assert out  # partial content present


def test_load_knowledge_bad_budget_falls_back(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "small")
    assert kb.load_knowledge([str(f)], None) != ""  # type: ignore[arg-type]
    assert kb.load_knowledge([str(f)], "bad") != ""  # type: ignore[arg-type]
    # huge budget clamped to 60000 (no crash, still fits)
    out = kb.load_knowledge([str(f)], 10**9)
    assert len(out) <= 60000


def test_load_knowledge_skips_binary_and_empty(tmp_path):
    empty = tmp_path / "empty.txt"
    binary = tmp_path / "blob.txt"
    good = tmp_path / "good.txt"
    _write(empty, "")
    binary.write_bytes(b"hello\x00world" + b"z" * 100)
    _write(good, "real content")
    out = kb.load_knowledge([str(empty), str(binary), str(good)], 12000)
    assert "real content" in out
    assert "empty.txt" not in out
    assert "blob.txt" not in out


def test_load_knowledge_skips_oversize(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (kb.MAX_FILE_BYTES + 10))
    assert kb.load_knowledge([str(f)], 12000) == ""


def test_read_invalid_utf8_does_not_crash(tmp_path):
    f = tmp_path / "latin.txt"
    f.write_bytes("café".encode("latin-1") + b" ok")
    out = kb.load_knowledge([str(f)], 12000)
    assert isinstance(out, str)


def test_knowledge_never_raises_on_garbage():
    assert isinstance(kb.load_knowledge([123, None, object()], 12000), str)  # type: ignore[list-item]
    assert isinstance(kb.list_knowledge_files([123, None, object()]), list)  # type: ignore[list-item]
