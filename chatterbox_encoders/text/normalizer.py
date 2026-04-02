"""
Text normalization utilities for Chatterbox.

Provides punctuation normalization and text cleanup functions.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def punc_norm(text: str) -> str:
    """
    Normalize punctuation in text.

    Quick cleanup function for punctuation from LLMs or
    containing chars not seen often in the dataset.

    Args:
        text: Input text string

    Returns:
        str: Normalized text

    Transformations:
        1. Capitalize first letter
        2. Remove multiple space chars
        3. Replace uncommon/LLM punctuation
        4. Normalize quotes
        5. Add full stop if no ending punctuation

    Examples:
        >>> punc_norm("hello... world")
        'Hello, world.'
        >>> punc_norm("TEST")
        'TEST.'
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/LLM punctuation
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punctuation
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    # Final whitespace cleanup
    text = " ".join(text.split())

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        str: Text with normalized whitespace

    Examples:
        >>> normalize_whitespace("Hello    world")
        'Hello world'
    """
    return " ".join(text.split())


def capitalize_first(text: str) -> str:
    """
    Capitalize first letter of text.

    Args:
        text: Input text

    Returns:
        str: Text with first letter capitalized

    Examples:
        >>> capitalize_first("hello")
        'Hello'
    """
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def add_ending_punctuation(
    text: str,
    punctuation: str = ".",
) -> str:
    """
    Add ending punctuation if not present.

    Args:
        text: Input text
        punctuation: Punctuation to add (default: ".")

    Returns:
        str: Text with ending punctuation

    Examples:
        >>> add_ending_punctuation("Hello")
        'Hello.'
        >>> add_ending_punctuation("Hello!", "?")
        'Hello!'
    """
    text = text.rstrip()
    if text and not text[-1] in {".", "!", "?", "-", ","}:
        text = text + punctuation
    return text
