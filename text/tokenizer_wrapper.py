"""
LLM-agnostic text tokenizer wrapper.

Accepts any tokenizer (HuggingFace, custom, etc.) for text tokenization.
"""

import logging
from typing import Union, Optional

import torch

logger = logging.getLogger(__name__)


def text_to_tokens(
    text: str,
    tokenizer: Optional[any] = None,
    **kwargs
) -> torch.Tensor:
    """
    Tokenize text using any LLM-agnostic tokenizer.

    This function accepts any tokenizer and converts text to token IDs.
    Supports HuggingFace tokenizers, custom tokenizers, and more.

    Args:
        text: Input text string to tokenize
        tokenizer: Any tokenizer with encode() method
            - transformers.AutoTokenizer: Use tokenizer(text, return_tensors="pt")
            - tokenizers.Tokenizer: Use tokenizer.encode(text)
            - Custom: Any object with encode() method
        **kwargs: Additional arguments passed to tokenizer

    Returns:
        torch.Tensor: Token IDs
            - Shape: (1, seq_len)
            - Ready for LLM input

    Raises:
        ValueError: If tokenizer is None

    Examples:
        >>> # HuggingFace tokenizer
        >>> from transformers import AutoTokenizer
        >>> hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokens = text_to_tokens("Hello world", tokenizer=hf_tokenizer)
        >>>
        >>> # Custom tokenizer
        >>> def my_tokenizer(text):
        ...     return [ord(c) for c in text]
        >>> tokens = text_to_tokens("Hello", tokenizer=my_tokenizer)

    Note:
        The tokenizer must have either:
        - A __call__ method (HuggingFace style)
        - An encode() method (custom style)
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")

    # Normalize text
    if isinstance(text, str):
        text = text.strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        text = " ".join(text.split())

    # Tokenize based on tokenizer type
    if hasattr(tokenizer, "__call__"):
        # HuggingFace style
        tokens = tokenizer(text, return_tensors="pt", **kwargs)
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
    elif hasattr(tokenizer, "encode"):
        # Custom tokenizer
        token_ids = tokenizer.encode(text, **kwargs)
        tokens = torch.IntTensor(token_ids).unsqueeze(0)
    else:
        raise TypeError(
            f"Tokenizer must have __call__ or encode method. Got {type(tokenizer)}"
        )

    return tokens
