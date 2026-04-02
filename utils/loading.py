"""
Model loading utilities for Chatterbox encoders.

This module provides safe model loading with proper device handling.
"""

import torch
import logging
from pathlib import Path
from typing import Union, Optional, Any
from safetensors.torch import load_file as safe_load_file

from chatterbox_encoders.utils.device import get_device

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    device: Union[str, torch.device, None] = None,
    strict: bool = True,
    weights_only: bool = True,
) -> dict:
    """
    Load model weights from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file (.safetensors or .pt)
        model: Model to load weights into (None to return weights dict)
        device: Target device (None for auto-detect)
        strict: Whether to strictly enforce state dict keys match
        weights_only: Only load weights (no other state)

    Returns:
        dict: State dictionary if model is None, otherwise empty dict

    Examples:
        >>> weights = load_model("model.safetensors")
        >>> model = MyModel()
        >>> load_model("model.safetensors", model=model, device="cuda")
    """
    checkpoint_path = Path(checkpoint_path)
    device_obj = get_device(device if device is not None else "auto")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Determine map_location based on device
    if device_obj.type == "cpu":
        map_location = torch.device("cpu")
    elif device_obj.type == "mps":
        map_location = torch.device("cpu")  # Load to CPU first for MPS
    else:
        map_location = None  # Keep original device

    # Load checkpoint
    if checkpoint_path.suffix == ".safetensors":
        state_dict = safe_load_file(str(checkpoint_path))
    elif checkpoint_path.suffix in [".pt", ".pth", ".ckpt"]:
        state_dict = torch.load(
            str(checkpoint_path),
            map_location=map_location,
            weights_only=weights_only
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")

    # Handle nested state dict (some models save with "model" key)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
        if isinstance(state_dict, list) and len(state_dict) > 0:
            state_dict = state_dict[0]

    # Load into model if provided
    if model is not None:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

        model = model.to(device_obj)
        model.eval()
        logger.info(f"Model loaded and moved to {device_obj}")
    else:
        logger.info(f"State dict loaded (not loading into model)")

    return state_dict if model is None else {}


def load_s3tokenizer(
    checkpoint_path: Union[str, Path] = None,
    device: Union[str, torch.device, None] = None,
) -> Any:
    """
    Load S3Tokenizer model.

    Args:
        checkpoint_path: Path to S3Tokenizer weights (None for default)
        device: Target device (None for auto-detect)

    Returns:
        S3Tokenizer instance

    Note:
        If checkpoint_path is None, will download from HuggingFace.
    """
    from s3tokenizer.model_v2 import S3TokenizerV2, ModelConfig

    device_obj = get_device(device if device is not None else "auto")

    if checkpoint_path is None:
        # Load default model from HuggingFace
        tokenizer = S3TokenizerV2(name="speech_tokenizer_v2_25hz")
        logger.info("Loaded default S3Tokenizer from HuggingFace")
    else:
        # Load from checkpoint
        tokenizer = S3TokenizerV2(name="speech_tokenizer_v2_25hz")
        load_model(checkpoint_path, model=tokenizer, device=device_obj)

    tokenizer = tokenizer.to(device_obj)
    tokenizer.eval()

    return tokenizer


def load_voice_encoder(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device, None] = None,
) -> Any:
    """
    Load VoiceEncoder model.

    Args:
        checkpoint_path: Path to VoiceEncoder weights
        device: Target device (None for auto-detect)

    Returns:
        VoiceEncoder instance
    """
    from chatterbox_encoders.audio.voice_encoder import VoiceEncoder

    device_obj = get_device(device if device is not None else "auto")

    ve = VoiceEncoder()
    load_model(checkpoint_path, model=ve, device=device_obj)

    ve = ve.to(device_obj)
    ve.eval()

    logger.info(f"VoiceEncoder loaded from {checkpoint_path}")

    return ve


def load_s3gen(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device, None] = None,
    meanflow: bool = False,
) -> Any:
    """
    Load S3Gen decoder model.

    Args:
        checkpoint_path: Path to S3Gen weights
        device: Target device (None for auto-detect)
        meanflow: Whether to use meanflow (distilled) version

    Returns:
        S3Gen instance
    """
    from chatterbox.models.s3gen import S3Gen

    device_obj = get_device(device if device is not None else "auto")

    s3gen = S3Gen(meanflow=meanflow)
    load_model(checkpoint_path, model=s3gen, device=device_obj)

    s3gen = s3gen.to(device_obj)
    s3gen.eval()

    logger.info(f"S3Gen loaded from {checkpoint_path}")

    return s3gen
