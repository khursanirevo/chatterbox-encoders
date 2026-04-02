"""
Voice Encoder module for speaker embedding extraction.

Extracts 256-dimensional speaker embeddings from audio using LSTM architecture.
"""

from typing import List, Union, Optional
from dataclasses import dataclass
import logging

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chatterbox_encoders.config.defaults import (
    VOICE_ENCODER_NUM_MELS,
    VOICE_ENCODER_SAMPLE_RATE,
    VOICE_ENCODER_HIDDEN_SIZE,
    VOICE_ENCODER_EMBED_SIZE,
    VOICE_ENCODER_NUM_LAYERS,
)

logger = logging.getLogger(__name__)


@dataclass
class VoiceEncConfig:
    """Configuration for VoiceEncoder."""
    normalized_mels: bool = True
    ve_partial_frames: int = 96
    sample_rate: int = VOICE_ENCODER_SAMPLE_RATE
    num_mels: int = VOICE_ENCODER_NUM_MELS
    ve_hidden_size: int = VOICE_ENCODER_HIDDEN_SIZE
    speaker_embed_size: int = VOICE_ENCODER_EMBED_SIZE
    num_layers: int = VOICE_ENCODER_NUM_LAYERS
    flatten_lstm_params: bool = True
    ve_final_relu: bool = False


def melspectrogram(
    wav: np.ndarray,
    hp: VoiceEncConfig,
) -> np.ndarray:
    """
    Extract mel spectrogram from audio.

    Args:
        wav: Audio waveform
        hp: VoiceEncoder configuration

    Returns:
        np.ndarray: Mel spectrogram (T, num_mels)
    """
    from librosa.filters import mel as librosa_mel_fn

    if isinstance(wav, np.ndarray):
        wav = torch.tensor(wav).float()

    if wav.dim() == 1:
        wav = wav[None, ]

    # Mel filterbank
    if not hasattr(melspectrogram, "mel_basis"):
        mel_basis = librosa_mel_fn(
            sr=hp.sample_rate,
            n_fft=400,
            n_mels=hp.num_mels,
        )
        melspectrogram.mel_basis = torch.from_numpy(mel_basis).float()

    spec = torch.stft(
        wav,
        n_fft=400,
        hop_length=160,
        window=torch.hann_window(400),
        return_complex=True,
    )
    magnitudes = spec.real ** 2 + spec.imag ** 2

    mel = melspectrogram.mel_basis @ magnitudes
    mel = torch.log(torch.clamp(mel, min=1e-10))

    return mel.squeeze(0).T  # (T, num_mels)


def pack(
    arrays: List[np.ndarray],
    seq_len: Optional[int] = None,
    pad_value: float = 0,
) -> torch.Tensor:
    """
    Pack a list of arrays into a single tensor by padding.

    Args:
        arrays: List of array-like objects
        seq_len: Target sequence length
        pad_value: Padding value

    Returns:
        torch.Tensor: Packed tensor (B, T, ...)
    """
    if seq_len is None:
        seq_len = max(len(arr) for arr in arrays)
    else:
        assert seq_len >= max(len(arr) for arr in arrays)

    if isinstance(arrays[0], list):
        arrays = [np.array(arr) for arr in arrays]

    device = None
    if isinstance(arrays[0], torch.Tensor):
        tensors = arrays
        device = tensors[0].device
    else:
        tensors = [torch.as_tensor(arr) for arr in arrays]

    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(
        packed_shape,
        pad_value,
        dtype=tensors[0].dtype,
        device=device,
    )

    for i, tensor in enumerate(tensors):
        packed_tensor[i, :tensor.size(0)] = tensor

    return packed_tensor


class VoiceEncoder(nn.Module):
    """
    Voice encoder for extracting speaker embeddings.

    Uses 3-layer LSTM architecture to extract 256-dimensional speaker embeddings
    from mel spectrograms. Embeddings are L2-normalized for cosine similarity.

    Args:
        hp: Configuration (VoiceEncConfig or None)

    Examples:
        >>> ve = VoiceEncoder()
        >>> ve.load_state_dict(torch.load("ve.safetensors"))
        >>> embedding = ve.embeds_from_wavs([audio], sample_rate=16000, as_spk=True)
        >>> embedding.shape
        (256,)
    """

    def __init__(self, hp: Optional[VoiceEncConfig] = None):
        super().__init__()

        self.hp = hp if hp is not None else VoiceEncConfig()

        # LSTM network
        self.lstm = nn.LSTM(
            self.hp.num_mels,
            self.hp.ve_hidden_size,
            num_layers=self.hp.num_layers,
            batch_first=True,
        )

        if self.hp.flatten_lstm_params:
            self.lstm.flatten_parameters()

        # Projection to speaker embedding dimension
        self.proj = nn.Linear(
            self.hp.ve_hidden_size,
            self.hp.speaker_embed_size,
        )

        # Cosine similarity scaling parameters
        self.similarity_weight = nn.Parameter(torch.tensor([10.]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]), requires_grad=True)

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    def forward(self, mels: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute embeddings of a batch of partial utterances.

        Args:
            mels: Batch of mel spectrograms
                - Shape: (B, T, M)
                - T: hp.ve_partial_frames (96 frames)
                - M: hp.num_mels (40 bands)

        Returns:
            torch.FloatTensor: Embeddings
                - Shape: (B, E)
                - E: hp.speaker_embed_size (256)
                - L2-normalized

        Examples:
            >>> ve = VoiceEncoder()
            >>> mels = torch.randn(2, 96, 40)  # Batch of 2
            >>> embeddings = ve(mels)
            >>> embeddings.shape
            torch.Size([2, 256])
        """
        if self.hp.normalized_mels:
            if mels.min() < 0 or mels.max() > 1:
                raise Exception(f"Mels outside [0, 1]. Min={mels.min()}, Max={mels.max()}")

        # LSTM processing
        _, (hidden, _) = self.lstm(mels)

        # Project to embedding dimension
        raw_embeds = self.proj(hidden[-1])

        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)

        # L2 normalize
        embeddings = raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

        return embeddings

    def embeds_from_mels(
        self,
        mels: Union[Tensor, List[np.ndarray]],
        as_spk: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Convenience function for deriving embeddings from mel spectrograms.

        Args:
            mels: Mel spectrograms
                - Can be: (B, T, M) tensor or list of (Ti, M) arrays
            as_spk: Whether to return speaker embedding (mean of utterances)

        Returns:
            np.ndarray: Embeddings
                - (B, E) if as_spk=False (utterance embeddings)
                - (E,) if as_spk=True (speaker embedding)

        Examples:
            >>> ve = VoiceEncoder()
            >>> mel1 = np.random.randn(100, 40)
            >>> mel2 = np.random.randn(150, 40)
            >>> spk_emb = ve.embeds_from_mels([mel1, mel2], as_spk=True)
            >>> spk_emb.shape
            (256,)
        """
        if isinstance(mels, List):
            mels = [np.asarray(mel) for mel in mels]
            assert all(m.shape[1] == mels[0].shape[1] for m in mels)
            mels = pack(mels)

        with torch.inference_mode():
            utt_embeds = self.forward(mels.to(self.device))

        if as_spk:
            utt_embeds_np = utt_embeds.cpu().numpy()
            spk_emb = np.mean(utt_embeds_np, axis=0)
            spk_emb = spk_emb / np.linalg.norm(spk_emb, 2)
            return spk_emb
        else:
            return utt_embeds.cpu().numpy()

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate: int,
        as_spk: bool = False,
        batch_size: int = 32,
        trim_top_db: Optional[float] = 20,
        **kwargs
    ) -> np.ndarray:
        """
        Extract speaker embeddings from audio waveforms.

        Args:
            wavs: List of audio waveforms
            sample_rate: Sample rate of audio
            as_spk: Return single speaker embedding (True) or utterance embeddings (False)
            batch_size: Batch size for processing
            trim_top_db: Trim silence threshold in dB

        Returns:
            np.ndarray: Speaker embeddings
                - (E,) if as_spk=True (single speaker embedding)
                - (B, E) if as_spk=False (utterance embeddings)

        Examples:
            >>> ve = VoiceEncoder()
            >>> audio1 = np.random.randn(16000)  # 1 second
            >>> audio2 = np.random.randn(16000)
            >>> spk_emb = ve.embeds_from_wavs([audio1, audio2], sample_rate=16000, as_spk=True)
            >>> spk_emb.shape
            (256,)
        """
        if sample_rate != self.hp.sample_rate:
            wavs = [
                librosa.resample(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate, res_type="kaiser_fast")
                for wav in wavs
            ]

        if trim_top_db:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]

        mels = [melspectrogram(wav, self.hp).T for w in wavs]

        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)
