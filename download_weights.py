#!/usr/bin/env python3
"""
Download Chatterbox model weights to the package.

Weights are downloaded from HuggingFace:
https://huggingface.co/khursanirevo/chatterbox-encoders-weights
"""

import sys
from huggingface_hub import hf_hub_download
from pathlib import Path

def main():
    # Create weights directory
    weights_dir = Path("chatterbox_encoders/weights")
    weights_dir.mkdir(exist_ok=True)

    # HuggingFace repository
    REPO_ID = "khursanirevo/chatterbox-encoders-weights"

    # Files to download
    files = [
        "s3gen.safetensors",  # S3Gen decoder (1007 MB)
        "ve.safetensors",     # Voice encoder (5.4 MB)
        "tokenizer.json",     # English tokenizer (25 KB)
    ]

    print("=" * 60)
    print("CHATTERBOX ENCODERS - WEIGHT DOWNLOAD")
    print("=" * 60)
    print(f"\nDownloading from: {REPO_ID}")
    print(f"Destination: {weights_dir}")
    print(f"\nTotal size: ~1.01 GB\n")

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Downloading {f}...")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                local_dir=str(weights_dir),
                local_dir_use_symlinks=False
            )

            # Show file size
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  ✅ Downloaded: {f} ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"  ❌ Failed to download {f}: {e}")
            print(f"\n  You can download manually from:")
            print(f"  https://huggingface.co/{REPO_ID}/tree/main\n")
            return 1

    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nAll weights saved to: {weights_dir.absolute()}")
    print("\nYou can now use the package:")
    print("  python -m chatterbox_encoders.pipelines.standalone <audio.wav>")

    return 0

if __name__ == "__main__":
    sys.exit(main())
