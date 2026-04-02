"""
Device management utilities for Chatterbox encoders.

This module provides functions for device detection and management.
"""

import torch
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Get a torch device, auto-detecting if requested.

    Args:
        device: Device specification. Can be:
            - "auto": Auto-detect (CUDA > MPS > CPU)
            - "cuda": Use CUDA
            - "mps": Use MPS (Apple Silicon)
            - "cpu": Use CPU
            - torch.device object

    Returns:
        torch.device: The selected device

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Force CUDA
        >>> device = get_device("cpu")  # Force CPU
    """
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Auto-detected device: {device}")
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS requested but not available, falling back to CPU")
        device = "cpu"

    return torch.device(device)


def auto_device() -> torch.device:
    """
    Auto-detect the best available device.

    Returns:
        torch.device: The best available device (CUDA > MPS > CPU)

    Examples:
        >>> device = auto_device()
        >>> tensor = torch.randn(10).to(device)
    """
    return get_device("auto")


def get_device_name(device: Union[str, torch.device, None] = None) -> str:
    """
    Get the string name of a device.

    Args:
        device: Device specification (None for auto-detect)

    Returns:
        str: Device name ("cuda", "mps", or "cpu")

    Examples:
        >>> get_device_name("cuda")
        'cuda'
        >>> get_device_name()  # Auto-detect
        'cuda'
    """
    device_obj = get_device(device if device is not None else "auto")
    return device_obj.type


def move_to_device(
    obj: torch.Tensor,
    device: Union[str, torch.device, None] = None
) -> torch.Tensor:
    """
    Move a tensor to the specified device.

    Args:
        obj: Tensor to move
        device: Target device (None for auto-detect)

    Returns:
        torch.Tensor: Tensor on the target device

    Examples:
        >>> tensor = torch.randn(10)
        >>> tensor = move_to_device(tensor)
        >>> tensor.device.type
        'cuda'
    """
    device_obj = get_device(device if device is not None else "auto")
    return obj.to(device_obj)


def get_device_memory(device: Union[str, torch.device, None] = None) -> dict:
    """
    Get memory information for a device.

    Args:
        device: Device to check (None for auto-detect)

    Returns:
        dict: Memory information with keys:
            - total: Total memory in GB
            - reserved: Reserved memory in GB
            - allocated: Allocated memory in GB
            - free: Free memory in GB

    Examples:
        >>> memory = get_device_memory("cuda")
        >>> print(f"Free: {memory['free']:.2f} GB")
    """
    device_obj = get_device(device if device is not None else "auto")

    if device_obj.type == "cuda":
        total = torch.cuda.get_device_properties(device_obj).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device_obj) / (1024**3)
        allocated = torch.cuda.memory_allocated(device_obj) / (1024**3)
        free = total - reserved
    else:
        total = reserved = allocated = free = 0.0

    return {
        "total": total,
        "reserved": reserved,
        "allocated": allocated,
        "free": free,
    }
