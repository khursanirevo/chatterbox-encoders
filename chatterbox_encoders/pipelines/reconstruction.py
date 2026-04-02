"""
Reconstruction pipeline for encode/decode validation.

Provides end-to-end pipeline for encoding audio to tokens and
decoding back to audio with quality metrics (MAE, MSE).
"""

import torch
import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Union, Tuple, Optional
import json
from datetime import datetime

from chatterbox_encoders.audio import S3Tokenizer
from chatterbox_encoders.audio.voice_encoder import VoiceEncoder
from chatterbox_encoders.audio.speaker_projector import SpeakerProjector
from chatterbox_encoders.audio.emotion import EmotionProjector
from chatterbox_encoders.audio.perceiver import PerceiverResampler
from chatterbox_encoders.utils.device import get_device
from chatterbox_encoders.utils.audio import load_audio, resample_audio
from chatterbox_encoders.utils.tokens import drop_invalid_tokens

logger = logging.getLogger(__name__)


class ReconstructionPipeline:
    """
    Complete pipeline for audio reconstruction validation.

    This class provides:
    1. Audio → Speech tokens encoding
    2. Speech tokens → Audio decoding (using S3Gen)
    3. Quality metrics calculation (MAE, MSE)

    Args:
        device: Target device (None for auto-detect)
        use_s3gen: Whether to use S3Gen for decoding (requires Chatterbox)

    Examples:
        >>> pipeline = ReconstructionPipeline()
        >>> mae, mse = pipeline.validate("test_audio.wav")
        >>> assert mae < 0.1 and mse < 0.1
    """

    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        use_s3gen: bool = True,
        load_tokenizer: bool = True,
    ):
        self.device = get_device(device if device is not None else "auto")
        self.use_s3gen = use_s3gen

        # Initialize components
        self.tokenizer = None
        self.voice_encoder = None
        self.s3gen = None

        # Load tokenizer by default for encoding
        if load_tokenizer:
            self.tokenizer = S3Tokenizer()
            logger.info("Loaded default S3Tokenizer")

        logger.info(f"ReconstructionPipeline initialized on {self.device}")

    def load_models(
        self,
        s3tokenizer_path: Optional[str] = None,
        voice_encoder_path: Optional[str] = None,
        s3gen_path: Optional[str] = None,
    ):
        """
        Load all required models.

        Args:
            s3tokenizer_path: Path to S3Tokenizer weights (None for default)
            voice_encoder_path: Path to VoiceEncoder weights
            s3gen_path: Path to S3Gen decoder weights
        """
        logger.info("Loading models...")

        # Load S3Tokenizer
        if s3tokenizer_path is None:
            # Use default from HuggingFace
            self.tokenizer = S3Tokenizer()
            logger.info("Loaded default S3Tokenizer")
        else:
            # TODO: Load from checkpoint
            pass

        # Load VoiceEncoder
        if voice_encoder_path is not None:
            self.voice_encoder = VoiceEncoder()
            from chatterbox_encoders.utils.loading import load_model

            load_model(voice_encoder_path, model=self.voice_encoder, device=self.device)
            logger.info(f"Loaded VoiceEncoder from {voice_encoder_path}")

        # Load S3Gen for decoding
        if self.use_s3gen and s3gen_path is not None:
            from chatterbox.models.s3gen import S3Gen

            self.s3gen = S3Gen()
            from chatterbox_encoders.utils.loading import load_model

            load_model(s3gen_path, model=self.s3gen, device=self.device)
            logger.info(f"Loaded S3Gen from {s3gen_path}")

    def encode(
        self,
        audio_path: Union[str, Path],
        sample_rate: int = 16000,
        max_duration: float = 10.0,
    ) -> dict:
        """
        Encode audio to speech tokens and features.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds

        Returns:
            dict: Encoded data with keys:
                - speech_tokens: Speech token IDs
                - token_length: Actual token length
                - audio_length: Audio duration
                - sample_rate: Sample rate

        Examples:
            >>> pipeline = ReconstructionPipeline()
            >>> encoded = pipeline.encode("test.wav")
            >>> tokens = encoded["speech_tokens"]
        """
        audio_path = Path(audio_path)

        logger.info(f"Encoding audio: {audio_path}")

        # Load audio
        audio, sr = load_audio(audio_path, sample_rate=sample_rate)

        # Trim to max duration
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            logger.info(f"Trimmed audio to {max_duration}s")

        # Tokenize
        tokens, lengths = self.tokenizer.forward([audio])

        encoded_data = {
            "speech_tokens": tokens[0].cpu().numpy().tolist(),
            "token_length": int(lengths[0].cpu().numpy()),
            "audio_length": len(audio) / sr,
            "sample_rate": sr,
            "encoding_time": datetime.now().isoformat(),
        }

        logger.info(
            f"Encoded: {encoded_data['token_length']} tokens, "
            f"{encoded_data['audio_length']:.2f}s"
        )

        return encoded_data

    def decode(
        self,
        speech_tokens: torch.Tensor,
        audio_ref: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """
        Decode speech tokens back to audio.

        Args:
            speech_tokens: Speech token IDs (B, T)
            audio_ref: Reference audio for speaker conditioning

        Returns:
            torch.Tensor: Reconstructed audio waveform

        Examples:
            >>> tokens = torch.tensor([[1, 2, 3, ...]])
            >>> wav = pipeline.decode(tokens)
        """
        if not self.use_s3gen:
            raise RuntimeError("S3Gen not loaded. Set use_s3gen=True and load s3gen_path.")

        logger.info(f"Decoding {speech_tokens.shape} tokens to audio...")

        # Prepare reference dict for S3Gen
        if audio_ref is not None:
            # TODO: Extract speaker embedding and mel features
            ref_dict = self._prepare_reference(audio_ref)
        else:
            raise ValueError("audio_ref is required for S3Gen decoding")

        # Decode
        with torch.inference_mode():
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
            )

        wav = wav.squeeze(0).cpu()

        logger.info(f"Decoded: {wav.shape}, {len(wav) / 24000:.2f}s @ 24kHz")

        return wav

    def _prepare_reference(self, audio_ref: Union[str, Path]) -> dict:
        """Prepare reference dictionary for S3Gen from audio."""
        # Load audio
        audio_24k, sr = load_audio(audio_ref, sample_rate=24000)

        # Trim to max length
        max_samples = int(10 * sr)  # 10 seconds max
        audio_24k = audio_24k[:max_samples]

        # Extract mel features
        from chatterbox_encoders.audio.mel_extractor import mel_spectrogram_24k

        mel_feat = mel_spectrogram_24k(audio_24k)

        # TODO: Extract speaker embedding
        # For now, use zeros as placeholder
        speaker_emb = torch.zeros(256)

        ref_dict = {
            "prompt_token": torch.zeros(1, 1, dtype=torch.long),
            "prompt_token_len": torch.tensor([1]),
            "prompt_feat": mel_feat.unsqueeze(0),
            "prompt_feat_len": None,
            "embedding": speaker_emb.unsqueeze(0),
        }

        return ref_dict

    def reconstruct(
        self,
        audio_path: Union[str, Path],
        save_reconstructed: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full reconstruction pipeline: encode → decode.

        Args:
            audio_path: Path to original audio
            save_reconstructed: Whether to save reconstructed audio
            output_path: Path to save reconstructed audio

        Returns:
            tuple: (reconstructed_audio, metrics)
                - reconstructed_audio: Reconstructed waveform
                - metrics: Dict with mae and mse

        Examples:
            >>> pipeline = ReconstructionPipeline()
            >>> wav, metrics = pipeline.reconstruct("test.wav")
            >>> print(f"MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")
        """
        logger.info(f"Reconstructing: {audio_path}")

        # Load original audio
        audio_orig, sr = load_audio(audio_path, sample_rate=24000)

        # Encode
        encoded = self.encode(audio_path)

        # Convert to tensor
        tokens = torch.tensor(encoded["speech_tokens"]).unsqueeze(0)

        # Decode
        wav_recon = self.decode(tokens, audio_ref=audio_path)

        # Calculate metrics
        metrics = self._calculate_metrics(audio_orig, wav_recon, sr)

        logger.info(
            f"Reconstruction metrics - MAE: {metrics['mae']:.4f}, "
            f"MSE: {metrics['mse']:.4f}"
        )

        # Save if requested
        if save_reconstructed and output_path:
            import torchaudio

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(str(output_path), wav_recon.unsqueeze(0), sr)
            logger.info(f"Saved reconstructed audio to: {output_path}")

        return wav_recon, metrics

    def _calculate_metrics(
        self,
        audio_orig: np.ndarray,
        audio_recon: torch.Tensor,
        sr: int,
    ) -> dict:
        """
        Calculate reconstruction quality metrics.

        Args:
            audio_orig: Original audio
            audio_recon: Reconstructed audio
            sr: Sample rate

        Returns:
            dict: Metrics with mae and mse
        """
        # Convert to numpy
        if isinstance(audio_recon, torch.Tensor):
            audio_recon = audio_recon.cpu().numpy()

        # Trim to same length
        min_len = min(len(audio_orig), len(audio_recon))
        audio_orig = audio_orig[:min_len]
        audio_recon = audio_recon[:min_len]

        # Calculate MAE
        mae = np.mean(np.abs(audio_orig - audio_recon))

        # Calculate MSE
        mse = np.mean((audio_orig - audio_recon) ** 2)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "length": min_len,
            "sample_rate": sr,
        }

    def validate(
        self,
        audio_path: Union[str, Path],
        max_mae: float = 0.1,
        max_mse: float = 0.1,
        save_reconstructed: bool = True,
    ) -> Tuple[float, float]:
        """
        Validate reconstruction quality against thresholds.

        Args:
            audio_path: Path to test audio
            max_mae: Maximum acceptable MAE (default: 0.1)
            max_mse: Maximum acceptable MSE (default: 0.1)
            save_reconstructed: Whether to save reconstructed audio

        Returns:
            tuple: (mae, mse) - Reconstruction metrics

        Raises:
            AssertionError: If metrics exceed thresholds

        Examples:
            >>> pipeline = ReconstructionPipeline()
            >>> mae, mse = pipeline.validate("test.wav")
            >>> assert mae < 0.1 and mse < 0.1
        """
        # Reconstruct
        output_path = Path(audio_path).stem + "_reconstructed.wav"
        wav_recon, metrics = self.reconstruct(
            audio_path,
            save_reconstructed=save_reconstructed,
            output_path=output_path,
        )

        mae = metrics["mae"]
        mse = metrics["mse"]

        # Validate
        assert mae < max_mae, f"MAE {mae:.4f} exceeds threshold {max_mae}"
        assert mse < max_mse, f"MSE {mse:.4f} exceeds threshold {max_mse}"

        logger.info(f"✅ Validation passed! MAE: {mae:.4f}, MSE: {mse:.4f}")

        return mae, mse
