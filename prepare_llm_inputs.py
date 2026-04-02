"""
Prepare all inputs for Chatterbox LLM generation manually.

This script demonstrates how to extract and prepare all the components
needed for Chatterbox LLM text-to-speech generation:

1. Text tokens (from text)
2. Speech tokens (from reference audio)
3. Speaker embeddings (from reference audio)
4. Speaker projection (256 → 1024 dims)

Usage:
    python prepare_llm_inputs.py --text "Hello world" --audio reference.wav
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import librosa
import json

from chatterbox_encoders.audio import S3Tokenizer, VoiceEncoder
from chatterbox_encoders.audio.speaker_projector import SpeakerProjector
from chatterbox_encoders.text.english_tokenizer import EnTokenizer
from chatterbox_encoders.text.normalizer import punc_norm
from chatterbox_encoders.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LLMInputPreparer:
    """
    Prepare all inputs needed for Chatterbox LLM generation.

    This class extracts and formats:
    - Text tokens (for LLM text input)
    - Speech tokens (for conditioning)
    - Speaker embeddings (for voice cloning)
    - Projected speaker tokens (for LLM conditioning)
    """

    def __init__(
        self,
        device: str = "auto",
        ve_checkpoint: str = None,
        tokenizer_path: str = None,
    ):
        """
        Initialize LLM input preparer.

        Args:
            device: Device to use (auto/cuda/mps/cpu)
            ve_checkpoint: Path to voice encoder checkpoint
            tokenizer_path: Path to text tokenizer
        """
        self.device = get_device(device if device != "auto" else "auto")

        logger.info(f"🔧 Initializing LLMInputPreparer on {self.device}")

        # Initialize S3Tokenizer (for speech tokens)
        logger.info("📝 Loading S3Tokenizer...")
        self.s3_tokenizer = S3Tokenizer()
        self.s3_tokenizer = self.s3_tokenizer.to(self.device)
        self.s3_tokenizer.eval()
        logger.info(f"   ✓ Vocab size: {self.s3_tokenizer.vocab_size}")
        logger.info(f"   ✓ Token rate: {self.s3_tokenizer.token_rate} tokens/sec")

        # Initialize VoiceEncoder (for speaker embeddings)
        logger.info("🎤 Loading VoiceEncoder...")
        self.voice_encoder = VoiceEncoder()
        if ve_checkpoint:
            from chatterbox_encoders.utils.loading import load_model
            load_model(ve_checkpoint, model=self.voice_encoder, device=self.device)
        self.voice_encoder = self.voice_encoder.to(self.device)
        self.voice_encoder.eval()
        logger.info(f"   ✓ Embedding size: 256")

        # Initialize SpeakerProjector (256 → 1024)
        logger.info("🔮 Loading SpeakerProjector...")
        self.speaker_projector = SpeakerProjector()
        self.speaker_projector = self.speaker_projector.to(self.device)
        self.speaker_projector.eval()
        logger.info(f"   ✓ 256 → 1024 projection")

        # Initialize Text Tokenizer
        logger.info("📄 Loading English tokenizer...")
        if tokenizer_path:
            vocab_path = tokenizer_path
        else:
            # Use default tokenizer from weights directory
            vocab_path = Path(__file__).parent / "chatterbox_encoders" / "weights" / "tokenizer.json"
            if not Path(vocab_path).exists():
                raise FileNotFoundError(
                    f"Tokenizer not found at {vocab_path}. "
                    f"Please run: python download_weights.py"
                )

        self.text_tokenizer = EnTokenizer(str(vocab_path))
        logger.info(f"   ✓ Vocab size: {self.text_tokenizer.tokenizer.get_vocab_size()}")

    def prepare_text_tokens(
        self,
        text: str,
        normalize: bool = True,
    ) -> dict:
        """
        Prepare text tokens for LLM input.

        Args:
            text: Input text string
            normalize: Whether to normalize text (punctuation, capitalization)

        Returns:
            dict: {
                "tokens": torch.Tensor (1, seq_len),
                "text": str (normalized),
                "length": int
            }
        """
        logger.info(f"📝 Preparing text tokens: '{text}'")

        # Normalize text
        if normalize:
            text = punc_norm(text)
            text = text.strip()
            if text and text[0].islower():
                text = text[0].upper() + text[1:]

        # Tokenize
        token_ids = self.text_tokenizer.encode(text)
        tokens = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        result = {
            "tokens": tokens,
            "text": text,
            "length": len(token_ids),
        }

        logger.info(f"   ✓ Tokens shape: {result['tokens'].shape}")
        logger.info(f"   ✓ Normalized: '{result['text']}'")

        return result

    def prepare_speech_tokens(
        self,
        audio_path: str,
        max_duration: float = 30.0,
    ) -> dict:
        """
        Prepare speech tokens from reference audio.

        Args:
            audio_path: Path to reference audio file
            max_duration: Maximum audio duration in seconds

        Returns:
            dict: {
                "tokens": torch.Tensor (1, num_tokens),
                "lengths": torch.Tensor (1,),
                "audio": np.ndarray (original audio),
                "sample_rate": int,
                "duration": float
            }
        """
        logger.info(f"🎵 Preparing speech tokens from: {audio_path}")

        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Trim to max duration
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            logger.info(f"   Trimmed to {max_duration}s")

        # Ensure float32
        audio = audio.astype(np.float32)

        # Tokenize
        with torch.no_grad():
            tokens, lengths = self.s3_tokenizer.forward([audio])

        result = {
            "tokens": tokens.to(self.device),
            "lengths": lengths.to(self.device),
            "audio": audio,
            "sample_rate": sr,
            "duration": len(audio) / sr,
        }

        logger.info(f"   ✓ Tokens shape: {result['tokens'].shape}")
        logger.info(f"   ✓ Duration: {result['duration']:.2f}s")

        return result

    def prepare_speaker_embedding(
        self,
        audio_path: str,
        as_spk: bool = True,
    ) -> dict:
        """
        Prepare speaker embedding from reference audio.

        Args:
            audio_path: Path to reference audio file
            as_spk: Whether to L2-normalize embedding

        Returns:
            dict: {
                "embedding": torch.Tensor (1, 256),
                "projected": torch.Tensor (1, 1024),
                "audio": np.ndarray,
            }
        """
        logger.info(f"🎤 Preparing speaker embedding from: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Ensure float32
        audio = audio.astype(np.float32)

        # Extract embedding
        with torch.no_grad():
            embedding = self.voice_encoder.embeds_from_wavs(
                [audio],
                sample_rate=16000,
                as_spk=as_spk,
            )

        # Convert numpy to tensor and move to device
        embedding = torch.from_numpy(embedding).unsqueeze(0).to(self.device)

        # Project to model dimension (256 → 1024)
        with torch.no_grad():
            projected = self.speaker_projector(embedding)

        result = {
            "embedding": embedding,
            "projected": projected,
            "audio": audio,
        }

        logger.info(f"   ✓ Embedding shape: {result['embedding'].shape}")
        logger.info(f"   ✓ Projected shape: {result['projected'].shape}")

        return result

    def prepare_all_inputs(
        self,
        text: str,
        audio_path: str,
        normalize_text: bool = True,
    ) -> dict:
        """
        Prepare all LLM inputs from text and reference audio.

        Args:
            text: Input text string
            audio_path: Path to reference audio file
            normalize_text: Whether to normalize text

        Returns:
            dict: Complete LLM inputs with all components
        """
        logger.info("="*60)
        logger.info("PREPARING COMPLETE LLM INPUTS")
        logger.info("="*60)

        # Prepare text tokens
        text_data = self.prepare_text_tokens(text, normalize=normalize_text)

        # Prepare speech tokens
        speech_data = self.prepare_speech_tokens(audio_path)

        # Prepare speaker embedding
        speaker_data = self.prepare_speaker_embedding(audio_path)

        # Combine all inputs
        inputs = {
            "text": text_data,
            "speech_tokens": speech_data,
            "speaker": speaker_data,
        }

        logger.info("="*60)
        logger.info("✅ ALL INPUTS PREPARED")
        logger.info("="*60)
        logger.info(f"Text tokens: {text_data['tokens'].shape}")
        logger.info(f"Speech tokens: {speech_data['tokens'].shape}")
        logger.info(f"Speaker embedding: {speaker_data['embedding'].shape}")
        logger.info(f"Speaker projected: {speaker_data['projected'].shape}")
        logger.info("="*60)

        return inputs

    def save_inputs(
        self,
        inputs: dict,
        output_path: str,
    ):
        """
        Save prepared inputs to file.

        Args:
            inputs: Prepared inputs dict from prepare_all_inputs
            output_path: Path to save inputs (.pt or .json)
        """
        output_path = Path(output_path)

        logger.info(f"💾 Saving inputs to: {output_path}")

        if output_path.suffix == ".pt":
            # Save as torch file (preserves tensors)
            save_dict = {
                "text_tokens": inputs["text"]["tokens"].cpu(),
                "speech_tokens": inputs["speech_tokens"]["tokens"].cpu(),
                "speech_lengths": inputs["speech_tokens"]["lengths"].cpu(),
                "speaker_embedding": inputs["speaker"]["embedding"].cpu(),
                "speaker_projected": inputs["speaker"]["projected"].cpu(),
                "text_normalized": inputs["text"]["text"],
                "audio_duration": inputs["speech_tokens"]["duration"],
            }
            torch.save(save_dict, output_path)

        elif output_path.suffix == ".json":
            # Save as JSON (for inspection)
            save_dict = {
                "text_normalized": inputs["text"]["text"],
                "text_tokens": inputs["text"]["tokens"].cpu().tolist(),
                "speech_tokens": inputs["speech_tokens"]["tokens"].cpu().tolist(),
                "speech_lengths": inputs["speech_tokens"]["lengths"].cpu().tolist(),
                "speaker_embedding": inputs["speaker"]["embedding"].cpu().tolist(),
                "speaker_projected": inputs["speaker"]["projected"].cpu().tolist(),
                "audio_duration": inputs["speech_tokens"]["duration"],
            }
            with open(output_path, "w") as f:
                json.dump(save_dict, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}. Use .pt or .json")

        logger.info(f"   ✓ Saved to: {output_path}")


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description="Prepare LLM inputs for Chatterbox generation"
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Input text to speak"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Reference audio file for speaker cloning"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="llm_inputs.pt",
        help="Output path for prepared inputs (default: llm_inputs.pt)"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["pt", "json"],
        default="pt",
        help="Output format (default: pt)"
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--ve-checkpoint",
        type=str,
        help="Path to voice encoder checkpoint"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize text"
    )

    args = parser.parse_args()

    # Initialize preparer
    preparer = LLMInputPreparer(
        device=args.device,
        ve_checkpoint=args.ve_checkpoint,
    )

    # Prepare all inputs
    inputs = preparer.prepare_all_inputs(
        text=args.text,
        audio_path=args.audio,
        normalize_text=not args.no_normalize,
    )

    # Save inputs
    output_path = f"{args.output}.{args.format}" if "." not in args.output else args.output
    preparer.save_inputs(inputs, output_path)

    print("\n✅ Done!")
    print(f"Inputs saved to: {output_path}")
    print("\nYou can now use these inputs for LLM generation.")


if __name__ == "__main__":
    main()
