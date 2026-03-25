"""Text normalization utilities for conformal prediction."""

import string


def remove_punctuation(input_string: str) -> str:
    """Remove all punctuation from a string."""
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)


def remove_articles(input_string: str) -> str:
    """Remove English articles (a, an, the) from a string."""
    articles = ['a', 'an', 'the']
    words = input_string.split()
    result = ' '.join(word for word in words if word.lower() not in articles)
    return result


def remove_duplicate_whitespace(input_string: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return ' '.join(input_string.split())


def normalize_text(text: str) -> str:
    """Apply full text normalization: lowercase, remove punctuation/articles, collapse whitespace."""
    return remove_duplicate_whitespace(remove_articles(remove_punctuation(text.lower())))


def process_list_of_strings(input_list: list) -> list:
    """Normalize a list of strings."""
    return [normalize_text(item) for item in input_list]


def process_list_of_dicts(input_list_of_dicts: list) -> list:
    """Normalize keys in a list of {text: count} dicts, merging duplicates.

    Args:
        input_list_of_dicts: List of dicts mapping response text to frequency/score.

    Returns:
        List of dicts with normalized keys and merged values.
    """
    processed_list_of_dicts = []
    for dictionary in input_list_of_dicts:
        processed_dict = {}
        for key, value in dictionary.items():
            processed_key = normalize_text(key)
            if processed_key in processed_dict:
                processed_dict[processed_key] += value
            else:
                processed_dict[processed_key] = value
        processed_list_of_dicts.append(processed_dict)
    return processed_list_of_dicts
