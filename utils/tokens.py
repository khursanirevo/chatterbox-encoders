"""
Token processing utilities for Chatterbox encoders.

This module provides token filtering and processing utilities.
"""

import torch
import numpy as np
import logging
from typing import Union, List

from chatterbox_encoders.config.constants import SPEECH_VOCAB_SIZE

logger = logging.getLogger(__name__)


def drop_invalid_tokens(
    tokens: torch.Tensor,
    vocab_size: int = SPEECH_VOCAB_SIZE,
) -> torch.Tensor:
    """
    Drop tokens that are >= vocab_size.

    Args:
        tokens: Input token tensor (1, T) or (B, T)
        vocab_size: Maximum valid token ID

    Returns:
        torch.Tensor: Filtered tokens

    Examples:
        >>> tokens = torch.tensor([[1, 2, 7000, 4, 6600]])
        >>> filtered = drop_invalid_tokens(tokens)
        >>> filtered
        tensor([[1, 2, 4]])
    """
    assert tokens.dim() <= 2 and tokens.shape[0] == 1, "Only batch size of 1 supported"

    original_length = tokens.shape[1]
    # Filter and preserve 2D shape
    mask = tokens[0] < vocab_size
    tokens = tokens[:, mask]

    new_length = tokens.shape[1]
    if new_length < original_length:
        logger.info(f"Dropped {original_length - new_length} invalid tokens")

    return tokens


def filter_special_tokens(
    tokens: torch.Tensor,
    special_tokens: List[int],
) -> torch.Tensor:
    """
    Filter out special tokens.

    Args:
        tokens: Input token tensor
        special_tokens: List of special token IDs to filter

    Returns:
        torch.Tensor: Filtered tokens

    Examples:
        >>> tokens = torch.tensor([[1, 2, 0, 3, 6561, 4]])
        >>> filtered = filter_special_tokens(tokens, [0, 6561])
        >>> filtered
        tensor([[1, 2, 3, 4]])
    """
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for special_token in special_tokens:
        mask = mask & (tokens != special_token)

    filtered = tokens[mask].reshape(1, -1)

    return filtered


def add_start_stop_tokens(
    tokens: torch.Tensor,
    start_token: int,
    stop_token: int,
) -> torch.Tensor:
    """
    Add start and stop tokens to a token sequence.

    Args:
        tokens: Input token tensor (1, T) or (B, T)
        start_token: Start token ID
        stop_token: Stop token ID

    Returns:
        torch.Tensor: Tokens with start/stop added

    Examples:
        >>> tokens = torch.tensor([[1, 2, 3]])
        >>> with_start_stop = add_start_stop_tokens(tokens, 255, 0)
        >>> with_start_stop
        tensor([[255, 1, 2, 3, 0]])
    """
    import torch.nn.functional as F

    # Add start token at beginning
    tokens = F.pad(tokens, (1, 0), value=start_token)

    # Add stop token at end
    tokens = F.pad(tokens, (0, 1), value=stop_token)

    return tokens


def truncate_tokens(
    tokens: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Truncate tokens to maximum length.

    Args:
        tokens: Input token tensor
        max_length: Maximum length

    Returns:
        torch.Tensor: Truncated tokens

    Examples:
        >>> tokens = torch.tensor([[1, 2, 3, 4, 5]])
        >>> truncated = truncate_tokens(tokens, max_length=3)
        >>> truncated
        tensor([[1, 2, 3]])
    """
    if tokens.shape[1] > max_length:
        logger.info(f"Truncating tokens from {tokens.shape[1]} to {max_length}")
        tokens = tokens[:, :max_length]

    return tokens


def pad_tokens(
    tokens: torch.Tensor,
    target_length: int,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Pad tokens to target length.

    Args:
        tokens: Input token tensor
        target_length: Target length
        pad_value: Padding value

    Returns:
        torch.Tensor: Padded tokens

    Examples:
        >>> tokens = torch.tensor([[1, 2, 3]])
        >>> padded = pad_tokens(tokens, target_length=5, pad_value=0)
        >>> padded
        tensor([[1, 2, 3, 0, 0]])
    """
    import torch.nn.functional as F

    if tokens.shape[1] < target_length:
        padding = target_length - tokens.shape[1]
        tokens = F.pad(tokens, (0, padding), value=pad_value)

    return tokens


def stack_tokens(
    tokens_list: List[torch.Tensor],
) -> torch.Tensor:
    """
    Stack a list of token tensors into a batch.

    Args:
        tokens_list: List of token tensors

    Returns:
        torch.Tensor: Stacked tokens (B, T)

    Examples:
        >>> tokens1 = torch.tensor([[1, 2, 3]])
        >>> tokens2 = torch.tensor([[4, 5]])
        >>> stacked = stack_tokens([tokens1, tokens2])
        >>> # Padded to same length internally
    """
    import torch.nn.functional as F

    # Find max length
    max_length = max(t.shape[1] for t in tokens_list)

    # Pad all to max length
    padded = []
    for tokens in tokens_list:
        if tokens.shape[1] < max_length:
            tokens = F.pad(tokens, (0, max_length - tokens.shape[1]), value=0)
        padded.append(tokens)

    # Stack
    stacked = torch.cat(padded, dim=0)

    return stacked


def get_token_length(
    tokens: torch.Tensor,
    pad_value: int = 0,
) -> int:
    """
    Get actual token length (excluding padding).

    Args:
        tokens: Token tensor (1, T)
        pad_value: Padding value

    Returns:
        int: Actual token length

    Examples:
        >>> tokens = torch.tensor([[1, 2, 3, 0, 0]])
        >>> length = get_token_length(tokens)
        >>> length
        3
    """
    # Find first padding token
    mask = tokens != pad_value
    lengths = mask.sum(dim=1)

    return lengths[0].item()
