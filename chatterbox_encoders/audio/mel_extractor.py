"""
Mel spectrogram extraction utilities.

Provides mel spectrogram extraction at different sample rates for various components.
"""

import logging
import numpy as np
import torch
import librosa
from librosa.filters import mel as librosa_mel_fn
from typing import Union, Optional

from chatterbox_encoders.config.constants import (
    S3_SR,
    S3GEN_SR,
)
from chatterbox_encoders.config.defaults import (
    MEL_16K_NUM_MELS,
    MEL_16K_SAMPLE_RATE,
    MEL_16K_N_FFT,
    MEL_16K_HOP,
    MEL_24K_NUM_MELS,
    MEL_24K_SAMPLE_RATE,
    MEL_24K_N_FFT,
    MEL_24K_HOP,
    MEL_24K_WIN,
    MEL_24K_FMIN,
    MEL_24K_FMAX,
    MEL_40K_NUM_MELS,
    MEL_40K_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1, clip_val: float = 1e-5) -> torch.Tensor:
    """Apply dynamic range compression."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Apply spectral normalization."""
    output = dynamic_range_compression_torch(magnitudes)
    return output


def _get_mel_basis(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: int = 0,
    fmax: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Get mel filterbank."""
    mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(mel).float().to(device)


def _get_hann_window(n_fft: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Get Hann window."""
    return torch.hann_window(n_fft).to(device)


# Global caches for mel basis and windows
_mel_basis_cache = {}
_hann_window_cache = {}


def mel_spectrogram_torch(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: int = 0,
    fmax: Optional[int] = None,
    center: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Compute mel spectrogram using torch (faster for batch processing).

    Args:
        audio: Audio tensor (B, T) or (T,)
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to center
        device: Device

    Returns:
        torch.Tensor: Mel spectrogram (B, n_mels, T)
    """
    # Ensure batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Get or create mel basis
    cache_key = f"{fmax}_{device}"
    if cache_key not in _mel_basis_cache:
        _mel_basis_cache[cache_key] = _get_mel_basis(
            sample_rate, n_fft, n_mels, fmin, fmax, device
        )
    mel_basis = _mel_basis_cache[cache_key]

    # Get or create Hann window
    if device not in _hann_window_cache:
        _hann_window_cache[device] = _get_hann_window(win_length, device)
    hann_window = _hann_window_cache[device]

    # Pad audio
    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    audio = audio.squeeze(1)

    # STFT
    spec = torch.view_as_real(
        torch.stft(
            audio,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    # Apply mel filterbank
    spec = torch.matmul(mel_basis, spec)

    # Log compression
    spec = spectral_normalize_torch(spec)

    return spec


def mel_spectrogram_16k(
    audio: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Extract 128-band mel spectrogram at 16kHz (for S3Tokenizer).

    Args:
        audio: Audio at 16kHz
        device: Target device

    Returns:
        torch.Tensor: Mel spectrogram (1, 128, T)

    Examples:
        >>> audio = np.random.randn(16000)  # 1 second
        >>> mel = mel_spectrogram_16k(audio)
        >>> mel.shape
        torch.Size([1, 128, ~100])
    """
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio).float()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel = mel_spectrogram_torch(
        audio,
        sample_rate=MEL_16K_SAMPLE_RATE,
        n_fft=MEL_16K_N_FFT,
        hop_length=MEL_16K_HOP,
        win_length=MEL_16K_N_FFT,
        n_mels=MEL_16K_NUM_MELS,
        fmin=0,
        fmax=None,
        center=False,
        device=device,
    )

    return mel


def mel_spectrogram_24k(
    audio: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Extract 80-band mel spectrogram at 24kHz (for S3Gen decoder).

    Args:
        audio: Audio at 24kHz
        device: Target device

    Returns:
        torch.Tensor: Mel spectrogram (1, 80, T)

    Examples:
        >>> audio = np.random.randn(24000)  # 1 second
        >>> mel = mel_spectrogram_24k(audio)
        >>> mel.shape
        torch.Size([1, 80, ~50])
    """
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio).float()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel = mel_spectrogram_torch(
        audio,
        sample_rate=MEL_24K_SAMPLE_RATE,
        n_fft=MEL_24K_N_FFT,
        hop_length=MEL_24K_HOP,
        win_length=MEL_24K_WIN,
        n_mels=MEL_24K_NUM_MELS,
        fmin=MEL_24K_FMIN,
        fmax=MEL_24K_FMAX,
        center=False,
        device=device,
    )

    return mel


def mel_spectrogram_40k(
    audio: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Extract 40-band mel spectrogram at 16kHz (for VoiceEncoder).

    Args:
        audio: Audio at 16kHz
        device: Target device

    Returns:
        torch.Tensor: Mel spectrogram (1, 40, T)

    Examples:
        >>> audio = np.random.randn(16000)  # 1 second
        >>> mel = mel_spectrogram_40k(audio)
        >>> mel.shape
        torch.Size([1, 40, ~100])
    """
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio).float()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Voice encoder uses 40 mels at 16kHz
    mel = mel_spectrogram_torch(
        audio,
        sample_rate=MEL_40K_SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=MEL_40K_NUM_MELS,
        fmin=0,
        fmax=None,
        center=False,
        device=device,
    )

    return mel
