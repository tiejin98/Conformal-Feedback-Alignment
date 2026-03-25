"""Tests for I/O utilities."""

import json
import pytest
from pathlib import Path
from cfa.utils.io import (
    save_json, load_json,
    save_jsonl, load_jsonl,
    save_pickle, load_pickle,
    save_text, load_text_as_literal,
    load_text_lines_as_literals,
)


class TestJsonIO:
    def test_roundtrip(self, tmp_path):
        data = {"key": "value", "list": [1, 2, 3]}
        path = tmp_path / "test.json"
        save_json(data, str(path))
        loaded = load_json(str(path))
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.json"
        save_json({"a": 1}, str(path))
        assert path.exists()


class TestJsonlIO:
    def test_roundtrip(self, tmp_path):
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        path = tmp_path / "test.jsonl"
        save_jsonl(data, str(path))
        loaded = load_jsonl(str(path))
        assert loaded == data

    def test_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        save_jsonl([], str(path))
        loaded = load_jsonl(str(path))
        assert loaded == []


class TestPickleIO:
    def test_roundtrip(self, tmp_path):
        data = {"key": [1, 2, 3], "nested": {"a": "b"}}
        path = tmp_path / "test.pkl"
        save_pickle(data, str(path))
        loaded = load_pickle(str(path))
        assert loaded == data


class TestTextIO:
    def test_save_and_load_literal(self, tmp_path):
        data = [{"a": 1, "b": 2}, {"c": 3}]
        path = tmp_path / "test.txt"
        save_text(data, str(path))
        loaded = load_text_as_literal(str(path))
        assert loaded == data

    def test_load_lines_as_literals(self, tmp_path):
        path = tmp_path / "lines.txt"
        with open(path, "w") as f:
            f.write("{'a': 1, 'b': 2}\n")
            f.write("{'c': 3, 'd': 4}\n")
        loaded = load_text_lines_as_literals(str(path))
        assert len(loaded) == 2
        assert loaded[0] == {"a": 1, "b": 2}
        assert loaded[1] == {"c": 3, "d": 4}
