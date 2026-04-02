#!/usr/bin/env python3
"""
Setup script to download weights and prepare standalone package.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("CHATTERBOX ENCODERS - STANDALONE SETUP")
    print("=" * 60)

    # Step 1: Download weights
    print("\n[1/3] Downloading model weights...")
    result = subprocess.run(
        [sys.executable, "download_weights.py"],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌ Failed to download weights")
        return 1

    # Step 2: Verify weights
    print("\n[2/3] Verifying weights...")
    weights_dir = Path("chatterbox_encoders/weights")
    required_files = [
        "s3gen.safetensors",
        "ve.safetensors",
        "tokenizer.json"
    ]

    for f in required_files:
        fpath = weights_dir / f
        if not fpath.exists():
            print(f"❌ Missing: {f}")
            return 1
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  ✅ {f}: {size_mb:.1f} MB")

    # Step 3: Test reconstruction
    print("\n[3/3] Testing reconstruction pipeline...")
    print("This may take a minute...")

    # Create test audio
    import numpy as np
    import soundfile as sf

    test_audio = np.random.randn(16000).astype(np.float32) * 0.3
    test_path = "test_setup.wav"
    sf.write(test_path, test_audio, 16000)

    # Test reconstruction
    try:
        from chatterbox_encoders.pipelines.standalone import StandaloneReconstructionPipeline

        pipeline = StandaloneReconstructionPipeline()
        audio_recon, metrics = pipeline.reconstruct(test_path)

        mae = metrics['mae']
        mse = metrics['mse']

        print(f"\n  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")

        if mae < 0.1 and mse < 0.1:
            print("  ✅ Reconstruction quality OK!")
        else:
            print("  ⚠️  Reconstruction quality below threshold")

    except Exception as e:
        print(f"❌ Reconstruction test failed: {e}")
        return 1

    # Cleanup
    Path(test_path).unlink(missing_ok=True)
    Path("test_setup_reconstructed.wav").unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nYour standalone package is ready!")
    print("All weights are in: chatterbox_encoders/weights/")
    print("\nUsage:")
    print("  python -m chatterbox_encoders.pipelines.standalone <audio.wav>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
