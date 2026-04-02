"""
Audio processing utilities for Chatterbox encoders.

This module provides audio loading, resampling, and preprocessing utilities.
"""

import torch
import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Union, Optional, Tuple

from chatterbox_encoders.utils.device import get_device

logger = logging.getLogger(__name__)


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono if True

    Returns:
        tuple: (audio_array, sample_rate)

    Examples:
        >>> audio, sr = load_audio("test.wav", sample_rate=16000)
        >>> audio.shape
        (samples,)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Loading audio: {audio_path}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=mono)

    logger.info(f"Loaded: {audio.shape}, {sr} Hz, {len(audio) / sr:.2f} seconds")

    return audio, sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    res_type: str = "kaiser_fast",
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        res_type: Resampling method (see librosa.resample)

    Returns:
        np.ndarray: Resampled audio

    Examples:
        >>> audio_16k = resample_audio(audio_24k, 24000, 16000)
    """
    if orig_sr == target_sr:
        return audio

    logger.info(f"Resampling: {orig_sr} Hz → {target_sr} Hz")

    audio_resampled = librosa.resample(
        audio,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type=res_type
    )

    return audio_resampled


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: float = 20,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.

    Args:
        audio: Input audio array
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence
        frame_length: Length of analysis frame
        hop_length: Number of samples between frames

    Returns:
        np.ndarray: Audio with silence trimmed

    Examples:
        >>> audio_trimmed = trim_silence(audio, sr=16000)
    """
    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )

    logger.info(f"Trimmed silence: {len(audio)} → {len(audio_trimmed)} samples")

    return audio_trimmed


def normalize_audio(
    audio: np.ndarray,
    method: str = "peak",
) -> np.ndarray:
    """
    Normalize audio.

    Args:
        audio: Input audio array
        method: Normalization method:
            - "peak": Peak normalize to [-1, 1]
            - "rms": RMS normalize

    Returns:
        np.ndarray: Normalized audio

    Examples:
        >>> audio_norm = normalize_audio(audio, method="peak")
    """
    if method == "peak":
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return audio


def pad_audio(
    audio: np.ndarray,
    target_length: int,
    mode: str = "constant",
) -> np.ndarray:
    """
    Pad or trim audio to target length.

    Args:
        audio: Input audio array
        target_length: Target length in samples
        mode: Padding mode ("constant", "reflect", etc.)

    Returns:
        np.ndarray: Padded/trimmed audio

    Examples:
        >>> audio_padded = pad_audio(audio, target_length=16000)
    """
    current_length = len(audio)

    if current_length < target_length:
        # Pad
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode=mode)
    elif current_length > target_length:
        # Trim
        audio = audio[:target_length]

    return audio


def ensure_sample_rate(
    audio: np.ndarray,
    sr: int,
    target_sr: int,
) -> Tuple[np.ndarray, int]:
    """
    Ensure audio is at target sample rate, resampling if necessary.

    Args:
        audio: Input audio array
        sr: Current sample rate
        target_sr: Target sample rate

    Returns:
        tuple: (audio_at_target_sr, target_sr)

    Examples:
        >>> audio_16k, sr = ensure_sample_rate(audio, 24000, 16000)
    """
    if sr == target_sr:
        return audio, sr

    logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz")
    audio_resampled = resample_audio(audio, sr, target_sr)

    return audio_resampled, target_sr


def audio_to_tensor(
    audio: np.ndarray,
    device: Union[str, torch.device, None] = None,
) -> torch.Tensor:
    """
    Convert numpy audio array to torch tensor on specified device.

    Args:
        audio: Input audio array
        device: Target device (None for auto-detect)

    Returns:
        torch.Tensor: Audio tensor on target device

    Examples:
        >>> tensor = audio_to_tensor(audio)
        >>> tensor.device.type
        'cuda'
    """
    device_obj = get_device(device if device is not None else "auto")

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    tensor = torch.from_numpy(audio).float().to(device_obj)

    return tensor


def tensor_to_audio(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert torch tensor to numpy audio array.

    Args:
        tensor: Audio tensor

    Returns:
        np.ndarray: Audio array

    Examples:
        >>> audio = tensor_to_audio(tensor)
    """
    return tensor.cpu().numpy()
