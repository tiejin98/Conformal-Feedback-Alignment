"""File I/O helpers for consistent data loading and saving across stages."""

import json
import pickle
import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_json(data, path: str, indent: int = 4):
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str):
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: list, path: str):
    """Save list of dicts as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved JSONL ({len(data)} items): {path}")


def load_jsonl(path: str) -> list:
    """Load JSON Lines file as list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_pickle(data, path: str):
    """Save data as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle: {path}")


def load_pickle(path: str):
    """Load data from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_text(data, path: str):
    """Save data as text (using repr for complex objects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(data))
        f.write("\n")
    logger.info(f"Saved text: {path}")


def load_text_as_literal(path: str):
    """Load a text file and parse it as a Python literal (list/dict)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return ast.literal_eval(content)


def load_text_lines_as_literals(path: str) -> list:
    """Load a text file where each line is a Python literal."""
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(ast.literal_eval(line))
    return result
