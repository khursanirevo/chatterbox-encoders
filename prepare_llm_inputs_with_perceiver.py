"""
Complete LLM input preparation with Perceiver Resampler.

This script demonstrates the full pipeline for preparing Chatterbox LLM inputs:
1. Text tokens (from text)
2. Speech tokens (from reference audio)
3. Speech token embeddings (learned embeddings)
4. Compressed speech tokens (32 tokens via Perceiver Resampler)
5. Speaker embeddings (from reference audio)
6. Speaker projection (256 → 1024 dims)

Usage:
    python prepare_llm_inputs_with_perceiver.py --text "Hello world" --audio reference.wav
"""

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import librosa
import json

from chatterbox_encoders.audio import S3Tokenizer, VoiceEncoder
from chatterbox_encoders.audio.speaker_projector import SpeakerProjector
from chatterbox_encoders.audio.perceiver import PerceiverResampler
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


class SpeechTokenEmbedding(nn.Module):
    """
    Learnable embedding layer for speech tokens.

    Converts speech token IDs (0-6560) to 1024-dimensional embeddings.
    """

    def __init__(self, vocab_size: int = 6561, embedding_dim: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        logger.info(f"SpeechTokenEmbedding: {vocab_size} → {embedding_dim}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert speech token IDs to embeddings.

        Args:
            tokens: Speech token IDs (B, T)

        Returns:
            Embeddings (B, T, 1024)
        """
        return self.embedding(tokens)


class CompleteLLMInputPreparer:
    """
    Prepare ALL inputs needed for Chatterbox LLM generation.

    This class extracts and formats:
    1. Text tokens (for LLM text input)
    2. Speech tokens (raw token IDs)
    3. Speech token embeddings (learned 1024-dim embeddings)
    4. Compressed speech tokens (32 tokens via Perceiver Resampler)
    5. Speaker embeddings (256-dim from VoiceEncoder)
    6. Speaker projected tokens (1024-dim for LLM conditioning)
    """

    def __init__(
        self,
        device: str = "auto",
        ve_checkpoint: str = None,
        tokenizer_path: str = None,
        load_perceiver: bool = True,
    ):
        """
        Initialize complete LLM input preparer.

        Args:
            device: Device to use (auto/cuda/mps/cpu)
            ve_checkpoint: Path to voice encoder checkpoint
            tokenizer_path: Path to text tokenizer
            load_perceiver: Whether to load Perceiver Resampler
        """
        self.device = get_device(device if device != "auto" else "auto")

        logger.info(f"🔧 Initializing CompleteLLMInputPreparer on {self.device}")

        # Initialize S3Tokenizer (for speech tokens)
        logger.info("📝 Loading S3Tokenizer...")
        self.s3_tokenizer = S3Tokenizer()
        self.s3_tokenizer = self.s3_tokenizer.to(self.device)
        self.s3_tokenizer.eval()
        logger.info(f"   ✓ Vocab size: {self.s3_tokenizer.vocab_size}")
        logger.info(f"   ✓ Token rate: {self.s3_tokenizer.token_rate} tokens/sec")

        # Initialize Speech Token Embedding
        logger.info("🎵 Loading SpeechTokenEmbedding...")
        self.speech_embedding = SpeechTokenEmbedding(
            vocab_size=self.s3_tokenizer.vocab_size,
            embedding_dim=1024,
        )
        self.speech_embedding = self.speech_embedding.to(self.device)
        self.speech_embedding.eval()
        logger.info(f"   ✓ {self.s3_tokenizer.vocab_size} → 1024")

        # Initialize Perceiver Resampler (compress to 32 tokens)
        if load_perceiver:
            logger.info("🗜️ Loading PerceiverResampler...")
            self.perceiver = PerceiverResampler(
                num_queries=32,
                query_dim=1024,
                embedding_dim=1024,
                num_heads=4,
            )
            self.perceiver = self.perceiver.to(self.device)
            self.perceiver.eval()
            logger.info(f"   ✓ Variable → 32 tokens")
        else:
            self.perceiver = None
            logger.info("   ⚠️ Perceiver Resampler disabled")

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
            normalize: Whether to normalize text

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

    def prepare_speech_tokens_with_embeddings(
        self,
        audio_path: str,
        max_duration: float = 30.0,
    ) -> dict:
        """
        Prepare speech tokens AND embeddings from reference audio.

        Args:
            audio_path: Path to reference audio file
            max_duration: Maximum audio duration in seconds

        Returns:
            dict: {
                "tokens": torch.Tensor (1, num_tokens),
                "embeddings": torch.Tensor (1, num_tokens, 1024),
                "compressed": torch.Tensor (1, 32, 1024) or None,
                "audio": np.ndarray,
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

        # Convert to embeddings
        with torch.no_grad():
            embeddings = self.speech_embedding(tokens)  # (1, T, 1024)

        # Compress with Perceiver Resampler (if loaded)
        compressed = None
        if self.perceiver is not None:
            with torch.no_grad():
                compressed = self.perceiver(embeddings)  # (1, 32, 1024)
            logger.info(f"   ✓ Compressed: {embeddings.shape} → {compressed.shape}")

        result = {
            "tokens": tokens.to(self.device),
            "embeddings": embeddings.to(self.device),
            "compressed": compressed.to(self.device) if compressed is not None else None,
            "audio": audio,
            "sample_rate": sr,
            "duration": len(audio) / sr,
        }

        logger.info(f"   ✓ Tokens shape: {result['tokens'].shape}")
        logger.info(f"   ✓ Embeddings shape: {result['embeddings'].shape}")
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
            embedding_np = self.voice_encoder.embeds_from_wavs(
                [audio],
                sample_rate=16000,
                as_spk=as_spk,
            )

        # Convert to tensor and move to device
        embedding = torch.from_numpy(embedding_np).unsqueeze(0).to(self.device)

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
        Prepare ALL LLM inputs from text and reference audio.

        Args:
            text: Input text string
            audio_path: Path to reference audio file
            normalize_text: Whether to normalize text

        Returns:
            dict: Complete LLM inputs with all components
        """
        logger.info("="*60)
        logger.info("PREPARING COMPLETE LLM INPUTS (WITH PERCEIVER)")
        logger.info("="*60)

        # Prepare text tokens
        text_data = self.prepare_text_tokens(text, normalize=normalize_text)

        # Prepare speech tokens + embeddings + compressed
        speech_data = self.prepare_speech_tokens_with_embeddings(audio_path)

        # Prepare speaker embedding
        speaker_data = self.prepare_speaker_embedding(audio_path)

        # Combine all inputs
        inputs = {
            "text": text_data,
            "speech": speech_data,
            "speaker": speaker_data,
        }

        logger.info("="*60)
        logger.info("✅ ALL INPUTS PREPARED")
        logger.info("="*60)
        logger.info(f"Text tokens: {text_data['tokens'].shape}")
        logger.info(f"Speech tokens: {speech_data['tokens'].shape}")
        logger.info(f"Speech embeddings: {speech_data['embeddings'].shape}")
        if speech_data['compressed'] is not None:
            logger.info(f"Speech compressed (32 tokens): {speech_data['compressed'].shape}")
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
                "text_normalized": inputs["text"]["text"],

                "speech_tokens": inputs["speech"]["tokens"].cpu(),
                "speech_embeddings": inputs["speech"]["embeddings"].cpu(),
                "speech_compressed": inputs["speech"]["compressed"].cpu() if inputs["speech"]["compressed"] is not None else None,

                "speaker_embedding": inputs["speaker"]["embedding"].cpu(),
                "speaker_projected": inputs["speaker"]["projected"].cpu(),

                "audio_duration": inputs["speech"]["duration"],
            }
            torch.save(save_dict, output_path)

        elif output_path.suffix == ".json":
            # Save as JSON (for inspection) - convert tensors to lists
            save_dict = {
                "text_normalized": inputs["text"]["text"],
                "text_tokens": inputs["text"]["tokens"].cpu().tolist(),

                "speech_tokens": inputs["speech"]["tokens"].cpu().tolist(),
                "speech_embeddings_shape": list(inputs["speech"]["embeddings"].shape),
                "speech_compressed": inputs["speech"]["compressed"].cpu().tolist() if inputs["speech"]["compressed"] is not None else None,

                "speaker_embedding": inputs["speaker"]["embedding"].cpu().tolist(),
                "speaker_projected": inputs["speaker"]["projected"].cpu().tolist(),

                "audio_duration": inputs["speech"]["duration"],
            }
            with open(output_path, "w") as f:
                json.dump(save_dict, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}. Use .pt or .json")

        logger.info(f"   ✓ Saved to: {output_path}")


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description="Prepare complete LLM inputs with Perceiver Resampler"
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
        default="llm_inputs_complete.pt",
        help="Output path for prepared inputs (default: llm_inputs_complete.pt)"
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
        "--no-perceiver",
        action="store_true",
        help="Disable Perceiver Resampler"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize text"
    )

    args = parser.parse_args()

    # Initialize preparer
    preparer = CompleteLLMInputPreparer(
        device=args.device,
        ve_checkpoint=args.ve_checkpoint,
        load_perceiver=not args.no_perceiver,
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
    print("\n📊 Summary:")
    print(f"  Text tokens: {inputs['text']['tokens'].shape}")
    print(f"  Speech tokens: {inputs['speech']['tokens'].shape}")
    print(f"  Speech embeddings: {inputs['speech']['embeddings'].shape}")
    if inputs['speech']['compressed'] is not None:
        print(f"  Speech compressed (32 tokens): {inputs['speech']['compressed'].shape}")
    print(f"  Speaker embedding: {inputs['speaker']['embedding'].shape}")
    print(f"  Speaker projected: {inputs['speaker']['projected'].shape}")
    print("\nYou can now use these inputs for LLM generation.")


if __name__ == "__main__":
    main()
