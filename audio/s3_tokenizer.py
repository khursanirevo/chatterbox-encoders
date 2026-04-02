"""
S3 Speech Tokenizer for Chatterbox encoders.

Converts audio waveforms to discrete speech tokens at 25 tokens/second.
Based on the S3Tokenizer from Chatterbox TTS.
"""

import logging
import math
from typing import List, Tuple, Union, Optional

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import (
    S3TokenizerV2,
    ModelConfig,
)

from chatterbox_encoders.config.constants import (
    S3_SR,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3_HOP,
    S3_TOKEN_HOP,
    N_FFT_S3,
    N_MELS_S3,
)

logger = logging.getLogger(__name__)


class S3Tokenizer(S3TokenizerV2):
    """
    Speech tokenizer that converts audio to discrete tokens.

    This tokenizer converts 16 kHz audio waveforms to speech tokens at a rate
    of 25 tokens per second with a vocabulary size of 6561 tokens.

    Architecture:
    - Mel spectrogram extraction (128 bands, 100 frames/sec)
    - 6-layer FSMN-enhanced transformer encoder
    - FSQ quantization: 1280-dim → 8-dim → single token ID
    - Rotary position embeddings
    - No learnable codebook (deterministic quantization)

    Args:
        name: Model name (default: "speech_tokenizer_v2_25hz")
        config: Model configuration

    Examples:
        >>> tokenizer = S3Tokenizer()
        >>> audio, sr = librosa.load("test.wav", sr=16000)
        >>> tokens, lengths = tokenizer.forward([audio])
        >>> tokens.shape
        torch.Size([1, num_tokens])
        >>> tokenizer.vocab_size
        6561
    """

    # Classes that will be ignored when loading state dict
    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str = "speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig(),
    ):
        super().__init__(name)

        self.n_fft = N_FFT_S3
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return SPEECH_VOCAB_SIZE

    @property
    def token_rate(self) -> int:
        """Get tokens per second."""
        return S3_TOKEN_RATE

    def pad(
        self,
        wavs: List[np.ndarray],
        sr: int
    ) -> List[torch.Tensor]:
        """
        Pad audio to multiple of 40ms (S3 runs at 25 token/sec).

        Args:
            wavs: List of audio arrays at same sample rate
            sr: Sample rate

        Returns:
            List of padded audio tensors

        Examples:
            >>> audio1 = np.random.randn(16000)
            >>> audio2 = np.random.randn(16000)
            >>> padded = tokenizer.pad([audio1, audio2], sr=16000)
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav,
                (0, intended_wav_len - wav.shape[-1]),
                mode="constant",
                value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(
        self,
        wavs: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """Prepare a list of audios for tokenization."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Tokenize audio waveforms to speech tokens.

        Args:
            wavs: 16 kHz speech audio (can be list of numpy arrays or single tensor)
            max_len: Maximum token length (25 token/sec, None for no limit)

        Returns:
            tuple: (speech_tokens, speech_token_lengths)
                - speech_tokens: (B, T) tensor of token IDs
                - speech_token_lengths: (B,) tensor of sequence lengths

        Note:
            Mel-spec has a hop size of 160 points (100 frame/sec).
            Please pad the waveform if longer sequence is needed.

        Examples:
            >>> tokenizer = S3Tokenizer()
            >>> audio = np.random.randn(16000)  # 1 second at 16kHz
            >>> tokens, lengths = tokenizer.forward([audio])
            >>> tokens.shape
            torch.Size([1, ~25])  # ~25 tokens for 1 second
        """
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)

        speech_tokens, speech_token_lens = self.quantize(
            mels.to(self.device),
            mel_lens.to(self.device)
        )

        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Compute log-Mel spectrogram of audio.

        Args:
            audio: Audio waveform
                - Shape: (*) (any shape)
                - 16 kHz
            padding: Number of zero samples to pad to the right

        Returns:
            torch.Tensor: Log mel spectrogram
                - Shape: (128, n_frames)
                - Normalized to [0, 1] range

        Examples:
            >>> audio = torch.randn(16000)  # 1 second
            >>> mel = tokenizer.log_mel_spectrogram(audio)
            >>> mel.shape
            torch.Size([128, ~100])  # ~100 frames
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec
