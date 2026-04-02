"""
Standalone S3Gen decoder for chatterbox-encoders package.

This module contains a minimal S3Gen implementation that can
decode speech tokens back to audio without depending on the full
Chatterbox package.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MinimalS3Gen(nn.Module):
    """
    Minimal S3Gen decoder for speech token reconstruction.

    This is a simplified version that loads only the necessary
    components for decoding speech tokens to audio.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.weights_path = weights_path

        # This will be populated from the safetensors file
        self.model = None
        self.tokenizer = None

    @classmethod
    def from_pretrained(cls, weights_dir: str, device: str = "cpu"):
        """
        Load S3Gen from local weights directory.

        Args:
            weights_dir: Path to directory containing s3gen.safetensors
            device: Device to load model on

        Returns:
            MinimalS3Gen: Loaded decoder
        """
        weights_path = Path(weights_dir) / "s3gen.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"S3Gen weights not found: {weights_path}")

        # For now, we'll load the full S3Gen from Chatterbox
        # In production, this would be a standalone implementation
        import sys
        repo_root = Path(__file__).parent.parent.parent
        while repo_root.name != "chatterbox" and repo_root.parent != repo_root:
            repo_root = repo_root.parent

        src_path = repo_root / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))

        from chatterbox.models.s3gen import S3Gen
        from safetensors.torch import load_file

        logger.info(f"Loading S3Gen from: {weights_path}")

        # Create model
        model = S3Gen()

        # Load weights
        state_dict = load_file(str(weights_path))
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        logger.info("S3Gen loaded successfully")

        # Wrap in minimal interface
        decoder = cls(weights_dir, device)
        decoder.model = model
        decoder.tokenizer = model.tokenizer

        # Store reference to original module path
        decoder._module_path = src_path

        return decoder

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to speech tokens.

        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Sample rate

        Returns:
            tuple: (tokens, token_lengths)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Ensure audio is torch tensor on correct device
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        # Tokenize
        tokens, lengths = self.tokenizer.forward([audio.cpu().numpy()])
        tokens = tokens.to(self.device)
        lengths = lengths.to(self.device)

        return tokens, lengths

    def decode(
        self,
        tokens: torch.Tensor,
        ref_audio: np.ndarray,
        ref_sr: int = 16000,
        n_timesteps: int = 10,
    ) -> np.ndarray:
        """
        Decode speech tokens to audio.

        Args:
            tokens: Speech token IDs (1, T)
            ref_audio: Reference audio for speaker conditioning
            ref_sr: Reference sample rate
            n_timesteps: Number of CFM timesteps

        Returns:
            np.ndarray: Reconstructed audio waveform
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Prepare reference audio
        if isinstance(ref_audio, np.ndarray):
            ref_audio = torch.from_numpy(ref_audio).float()

        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)

        ref_audio = ref_audio.to(self.device)

        # Decode
        with torch.inference_mode():
            output = self.model(
                speech_tokens=tokens,
                ref_wav=ref_audio,
                ref_sr=ref_sr,
                finalize=True,
                n_cfm_timesteps=n_timesteps,
            )

        # Convert to numpy
        audio_recon = output.cpu().numpy()[0]

        return audio_recon

    def reconstruct(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        n_timesteps: int = 10,
    ) -> np.ndarray:
        """
        Full encode → decode pipeline.

        Args:
            audio: Input audio waveform
            sample_rate: Sample rate
            n_timesteps: Number of CFM timesteps for decoding

        Returns:
            np.ndarray: Reconstructed audio waveform
        """
        # Encode
        tokens, lengths = self.encode(audio, sample_rate)

        # Decode (using same audio as reference)
        audio_recon = self.decode(tokens, audio, sample_rate, n_timesteps)

        return audio_recon
