"""Tests for text processing utilities."""

import pytest
from cfa.utils.text_processing import (
    remove_punctuation,
    remove_articles,
    remove_duplicate_whitespace,
    normalize_text,
    process_list_of_strings,
    process_list_of_dicts,
)


class TestRemovePunctuation:
    def test_basic(self):
        assert remove_punctuation("hello, world!") == "hello world"

    def test_no_punctuation(self):
        assert remove_punctuation("hello world") == "hello world"

    def test_all_punctuation(self):
        assert remove_punctuation("!@#$%") == ""

    def test_empty(self):
        assert remove_punctuation("") == ""


class TestRemoveArticles:
    def test_removes_a(self):
        assert remove_articles("this is a test") == "this is test"

    def test_removes_an(self):
        assert remove_articles("this is an example") == "this is example"

    def test_removes_the(self):
        assert remove_articles("the cat sat") == "cat sat"

    def test_removes_all(self):
        assert remove_articles("a the an") == ""

    def test_case_insensitive(self):
        assert remove_articles("The cat and A dog") == "cat and dog"

    def test_no_articles(self):
        assert remove_articles("hello world") == "hello world"


class TestRemoveDuplicateWhitespace:
    def test_multiple_spaces(self):
        assert remove_duplicate_whitespace("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        assert remove_duplicate_whitespace("hello\t\nworld") == "hello world"

    def test_leading_trailing(self):
        assert remove_duplicate_whitespace("  hello  ") == "hello"

    def test_already_clean(self):
        assert remove_duplicate_whitespace("hello world") == "hello world"


class TestNormalizeText:
    def test_full_normalization(self):
        result = normalize_text("The Cat, sat on A mat!")
        assert result == "cat sat on mat"

    def test_preserves_content_words(self):
        result = normalize_text("summarize the following text")
        assert result == "summarize following text"


class TestProcessListOfStrings:
    def test_basic(self):
        result = process_list_of_strings(["The Cat!", "A Dog."])
        assert result == ["cat", "dog"]


class TestProcessListOfDicts:
    def test_merges_normalized_keys(self):
        input_data = [{"The cat!": 3, "the cat": 2}]
        result = process_list_of_dicts(input_data)
        assert len(result) == 1
        assert "cat" in result[0]
        assert result[0]["cat"] == 5  # 3 + 2

    def test_multiple_dicts(self):
        input_data = [
            {"hello!": 1, "world.": 2},
            {"A test": 5},
        ]
        result = process_list_of_dicts(input_data)
        assert len(result) == 2
        assert result[0] == {"hello": 1, "world": 2}
        assert result[1] == {"test": 5}

    def test_empty_list(self):
        assert process_list_of_dicts([]) == []
