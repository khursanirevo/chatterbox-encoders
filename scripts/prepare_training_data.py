"""
Helper script to prepare labels.json for training TextToAudioEmbedding.

Scans a directory for audio files and creates a labels.json template
that you can fill in with your text labels.

Usage:
    python scripts/prepare_training_data.py --data-dir data/my_audio/
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare labels.json for training")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for labels.json (default: data-dir/labels.json)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="wav,mp3,flac,m4a,ogg",
        help="Audio file extensions to include (comma-separated)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "labels.json"

    # Find all audio files
    extensions = args.extensions.split(",")
    audio_files = []

    for ext in extensions:
        audio_files.extend(data_dir.glob(f"*.{ext}"))
        audio_files.extend(data_dir.glob(f"*.{ext.upper()}"))

    # Remove duplicates and sort
    audio_files = sorted(set(audio_files))

    if not audio_files:
        print(f"⚠️  No audio files found in {data_dir}")
        print(f"   Looking for extensions: {args.extensions}")
        return

    # Create labels template
    labels = {}
    for audio_file in audio_files:
        # Use just the filename as the key
        labels[audio_file.name] = ""  # Empty string for you to fill in

    # Save labels template
    with output_path.open("w") as f:
        json.dump(labels, f, indent=2)

    print(f"✅ Created labels template: {output_path}")
    print(f"   Found {len(audio_files)} audio files")
    print("\n📝 Next steps:")
    print(f"   1. Edit {output_path}")
    print("   2. Fill in text labels for each audio file")
    print(f"   3. Run training: python scripts/train_text_encoder.py --data-dir {data_dir}")


if __name__ == "__main__":
    main()
