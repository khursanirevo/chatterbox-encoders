"""
Manual reconstruction using separate components.

This script shows how to use S3Tokenizer and S3GenDecoder independently
for audio → tokens → audio reconstruction.
"""

from pathlib import Path
import random

import librosa
import numpy as np
import soundfile as sf
import torch

from chatterbox_encoders.audio import S3Tokenizer
from chatterbox_encoders.audio.decoder import S3GenDecoder


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def manual_reconstruction(
    audio_path: str,
    weights_dir: str = None,
    device: str = "auto",
    n_cfm_timesteps: int = 10,
    seed: int = 42,
):
    """
    Manual reconstruction pipeline with separate components.

    Args:
        audio_path: Path to input audio file
        weights_dir: Path to weights directory (default: chatterbox_encoders/weights/)
        device: Device to use (auto/cuda/mps/cpu)
        n_cfm_timesteps: Number of CFM timesteps for decoding
        seed: Random seed for reproducibility

    Returns:
        tuple: (tokens, audio_reconstructed, metrics)
    """

    # Set seed for reproducibility
    set_seed(seed)

    # ===== STEP 1: Initialize Components =====

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"📱 Using device: {device}")

    # Initialize decoder first (it contains the tokenizer)
    print("🔧 Loading S3GenDecoder decoder...")
    if weights_dir is None:
        weights_dir = Path(__file__).parent / "chatterbox_encoders" / "weights"

    decoder = S3GenDecoder.from_pretrained(
        weights_dir=str(weights_dir),
        device=device
    )
    print(f"   ✓ Decoder loaded from {weights_dir}")

    # Use the decoder's internal tokenizer (has trained weights!)
    print("🔧 Using decoder's internal tokenizer...")
    tokenizer = decoder.tokenizer
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")
    print(f"   ✓ Token rate: {tokenizer.token_rate} tokens/sec")

    # ===== STEP 2: Load Audio =====

    print(f"\n🎵 Loading audio: {audio_path}")

    # Load at 24kHz for metrics comparison (original)
    audio_24k, sr_24k = librosa.load(audio_path, sr=24000, mono=True)

    # Trim to 10 seconds max
    max_samples_24k = 10 * sr_24k
    if len(audio_24k) > max_samples_24k:
        audio_24k = audio_24k[:max_samples_24k]

    # Load at 16kHz for tokenization (direct load from disk)
    audio_16k_encode, sr_16k = librosa.load(audio_path, sr=16000, mono=True)
    max_samples_16k = 10 * sr_16k
    if len(audio_16k_encode) > max_samples_16k:
        audio_16k_encode = audio_16k_encode[:max_samples_16k]

    # Resample 24k to 16k for reference audio (used by decoder)
    audio_16k_ref = librosa.resample(audio_24k, orig_sr=24000, target_sr=16000)

    print(f"   ✓ Audio length: {len(audio_16k_encode) / sr_16k:.2f} seconds")

    # ===== STEP 3: Encode (Audio → Tokens) =====

    print("\n🔢 Encoding audio to speech tokens...")

    # Ensure float32
    audio_16k_encode = audio_16k_encode.astype(np.float32)

    # Tokenize (use direct 16kHz load)
    with torch.no_grad():
        tokens, lengths = tokenizer.forward([audio_16k_encode])

    print(f"   ✓ Tokens shape: {tokens.shape}")
    print(f"   ✓ Token length: {lengths[0].item()} tokens")
    print(f"   ✓ Expected: ~{len(audio_16k_encode) / sr_16k * tokenizer.token_rate:.0f} tokens")

    # ===== STEP 4: Decode (Tokens → Audio) =====

    print(f"\n🔊 Decoding tokens to audio (n_timesteps={n_cfm_timesteps})...")

    # Use resampled 16kHz audio as reference for speaker conditioning
    audio_recon = decoder.decode(
        tokens=tokens,
        ref_audio=audio_16k_ref,
        ref_sr=16000,
        n_timesteps=n_cfm_timesteps,
    )

    print(f"   ✓ Reconstructed audio shape: {audio_recon.shape}")
    print("   ✓ Sample rate: 24000 Hz")

    # ===== STEP 5: Calculate Metrics =====

    print("\n📊 Calculating reconstruction metrics...")

    # Trim to same length for comparison
    min_len = min(len(audio_24k), len(audio_recon))
    audio_24k_trimmed = audio_24k[:min_len]
    audio_recon_trimmed = audio_recon[:min_len]

    mae = np.mean(np.abs(audio_24k_trimmed - audio_recon_trimmed))
    mse = np.mean((audio_24k_trimmed - audio_recon_trimmed) ** 2)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "length_samples": min_len,
        "length_seconds": min_len / 24000,
    }

    print(f"   ✓ MAE: {mae:.4f} {'✅' if mae < 0.1 else '❌'} (threshold: < 0.1)")
    print(f"   ✓ MSE: {mse:.4f} {'✅' if mse < 0.1 else '❌'} (threshold: < 0.1)")

    # ===== STEP 6: Save Output =====

    output_path = Path(audio_path).stem + "_manual_reconstructed.wav"
    sf.write(output_path, audio_recon, 24000)
    print(f"\n💾 Saved to: {output_path}")

    return tokens, audio_recon, metrics


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Manual audio reconstruction using separate components"
    )
    parser.add_argument(
        "audio_file",
        help="Path to input audio file"
    )
    parser.add_argument(
        "--n-timesteps",
        "-n",
        type=int,
        default=10,
        help="Number of CFM timesteps for decoding (default: 10)"
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for reconstructed audio"
    )

    args = parser.parse_args()

    try:
        tokens, audio_recon, metrics = manual_reconstruction(
            args.audio_file,
            n_cfm_timesteps=args.n_timesteps,
            device=args.device,
            seed=args.seed
        )

        # Save to custom output if specified
        if args.output:
            import soundfile as sf
            sf.write(args.output, audio_recon, 24000)
            print(f"\n💾 Saved to: {args.output}")

        print("\n" + "="*60)
        print("MANUAL RECONSTRUCTION COMPLETE")
        print("="*60)
        print(f"CFM Timesteps: {args.n_timesteps}")
        print(f"Tokens shape: {tokens.shape}")
        print(f"Audio shape: {audio_recon.shape}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Did you run: python download_weights.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
