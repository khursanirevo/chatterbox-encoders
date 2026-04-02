"""
Standalone reconstruction pipeline for chatterbox-encoders.

This pipeline is completely independent and uses only the local
weights included in the package.
"""

import sys
import logging
import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)


class StandaloneReconstructionPipeline:
    """
    Complete reconstruction pipeline using local weights.

    This is a standalone implementation that does NOT depend on
    the Chatterbox package. It uses only the weights included in
    chatterbox_encoders/weights/.

    Example:
        >>> pipeline = StandaloneReconstructionPipeline()
        >>> mae, mse = pipeline.validate("test_audio.wav")
        >>> assert mae < 0.1 and mse < 0.1
    """

    def __init__(
        self,
        weights_dir: Union[str, Path] = None,
        device: str = "auto",
        n_cfm_timesteps: int = 10,
    ):
        """
        Initialize pipeline.

        Args:
            weights_dir: Path to weights directory (default: chatterbox_encoders/weights/)
            device: Device to use (auto/cuda/mps/cpu)
            n_cfm_timesteps: Number of CFM timesteps for S3Gen
        """
        # Determine weights directory
        if weights_dir is None:
            weights_dir = Path(__file__).parent.parent / "weights"
        else:
            weights_dir = Path(weights_dir)

        self.weights_dir = weights_dir
        self.n_cfm_timesteps = n_cfm_timesteps

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        logger.info(f"Weights directory: {self.weights_dir}")

        # Verify weights exist
        s3gen_path = self.weights_dir / "s3gen.safetensors"
        if not s3gen_path.exists():
            raise FileNotFoundError(
                f"S3Gen weights not found at: {s3gen_path}\n"
                f"Please run: python download_weights.py"
            )

        # Load decoder
        self.decoder = self._load_decoder()

    def _load_decoder(self):
        """Load S3Gen decoder from local weights."""
        from chatterbox_encoders.audio.decoder import S3GenDecoder

        logger.info("Loading S3Gen decoder...")
        decoder = S3GenDecoder.from_pretrained(
            str(self.weights_dir),
            device=self.device
        )
        logger.info("Decoder loaded!")

        return decoder

    def encode(
        self,
        audio_path: Union[str, Path],
        sample_rate: int = 16000,
        max_duration: float = 10.0,
    ) -> dict:
        """
        Encode audio to speech tokens.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            max_duration: Maximum duration in seconds

        Returns:
            dict: Encoded data with tokens and metadata
        """
        audio_path = Path(audio_path)

        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

        # Trim to max duration
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Encode
        tokens, lengths = self.decoder.encode(audio, sr)

        return {
            "speech_tokens": tokens,
            "token_length": int(lengths[0]),
            "audio_length": len(audio) / sr,
            "sample_rate": sr,
            "audio": audio,
        }

    def decode(
        self,
        tokens: torch.Tensor,
        ref_audio: np.ndarray,
        ref_sr: int = 16000,
    ) -> np.ndarray:
        """
        Decode speech tokens to audio.

        Args:
            tokens: Speech token IDs (1, T)
            ref_audio: Reference audio for speaker conditioning
            ref_sr: Reference sample rate

        Returns:
            np.ndarray: Reconstructed audio
        """
        return self.decoder.decode(
            tokens=tokens,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            n_timesteps=self.n_cfm_timesteps,
        )

    def reconstruct(
        self,
        audio_path: Union[str, Path],
        save_reconstructed: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Full reconstruction pipeline: encode → decode.

        Args:
            audio_path: Path to original audio
            save_reconstructed: Whether to save reconstructed audio
            output_path: Path to save reconstructed audio

        Returns:
            tuple: (reconstructed_audio, metrics)
        """
        audio_path = Path(audio_path)
        logger.info(f"Reconstructing: {audio_path}")

        # Load original audio
        audio_orig, sr = librosa.load(str(audio_path), sr=24000, mono=True)

        # Trim to max duration
        max_samples = int(10 * sr)
        if len(audio_orig) > max_samples:
            audio_orig = audio_orig[:max_samples]

        # Also load at 16kHz for encoding
        audio_16k = librosa.resample(audio_orig, orig_sr=24000, target_sr=16000)

        # Encode
        encoded = self.encode(audio_path, sample_rate=16000, max_duration=10.0)
        tokens = encoded["speech_tokens"]

        # Decode
        audio_recon = self.decode(tokens, audio_16k, ref_sr=16000)

        # Calculate metrics
        metrics = self._calculate_metrics(audio_orig, audio_recon, sr)

        logger.info(
            f"Reconstruction metrics - MAE: {metrics['mae']:.4f}, "
            f"MSE: {metrics['mse']:.4f}"
        )

        # Save if requested
        if save_reconstructed:
            if output_path is None:
                output_path = Path(audio_path).stem + "_reconstructed.wav"
            else:
                output_path = Path(output_path)

            import soundfile as sf
            sf.write(str(output_path), audio_recon, 24000)
            logger.info(f"Saved reconstructed audio to: {output_path}")

        return audio_recon, metrics

    def _calculate_metrics(
        self,
        audio_orig: np.ndarray,
        audio_recon: np.ndarray,
        sr: int,
    ) -> dict:
        """Calculate reconstruction quality metrics."""
        # Trim to same length
        min_len = min(len(audio_orig), len(audio_recon))
        audio_orig = audio_orig[:min_len]
        audio_recon = audio_recon[:min_len]

        # Calculate metrics
        mae = np.mean(np.abs(audio_orig - audio_recon))
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
    ) -> Tuple[float, float]:
        """
        Validate reconstruction quality against thresholds.

        Args:
            audio_path: Path to test audio
            max_mae: Maximum acceptable MAE
            max_mse: Maximum acceptable MSE

        Returns:
            tuple: (mae, mse)

        Raises:
            AssertionError: If metrics exceed thresholds
        """
        wav_recon, metrics = self.reconstruct(audio_path)

        mae = metrics["mae"]
        mse = metrics["mse"]

        # Validate
        assert mae < max_mae, f"MAE {mae:.4f} exceeds threshold {max_mae}"
        assert mse < max_mse, f"MSE {mse:.4f} exceeds threshold {max_mse}"

        logger.info(f"✅ Validation passed! MAE: {mae:.4f}, MSE: {mse:.4f}")

        return mae, mse


def main():
    """Test standalone pipeline."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python standalone.py <audio_file>")
        return 1

    audio_path = sys.argv[1]

    try:
        # Initialize pipeline
        pipeline = StandaloneReconstructionPipeline()

        # Validate
        mae, mse = pipeline.validate(audio_path)

        print(f"\n{'='*60}")
        print(f"STANDALONE RECONSTRUCTION RESULTS")
        print(f"{'='*60}")
        print(f"MAE:  {mae:.8f}")
        print(f"MSE:  {mse:.8f}")
        print(f"Threshold: < 0.1")
        print(f"\nMAE < 0.1:  {'✅ PASS' if mae < 0.1 else '❌ FAIL'}")
        print(f"MSE < 0.1:  {'✅ PASS' if mse < 0.1 else '❌ FAIL'}")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        logger.error(f"❌ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
