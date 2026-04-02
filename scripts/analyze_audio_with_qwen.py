"""
Utility script to analyze audio files with Qwen3-Omni.

Extracts rich text analysis including emotion, profile, mood, speed,
prosody, pitch_timbre, style, notes, and caption.
"""

import argparse
import json
import logging
from pathlib import Path

from chatterbox_encoders.text_analysis import QwenOmniAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze audio with Qwen3-Omni")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--format-for-encoder",
        action="store_true",
        help="Format output for text encoder input",
    )

    args = parser.parse_args()

    # Initialize analyzer
    logger.info("🎤 Initializing Qwen3-Omni analyzer...")
    analyzer = QwenOmniAnalyzer(device=args.device)

    # Analyze audio
    logger.info(f"📂 Analyzing: {args.audio_path}")
    analysis = analyzer.analyze_from_file(args.audio_path)

    # Format output
    if args.format_for_encoder:
        output = analyzer.format_for_encoder(analysis)
    else:
        output = analysis

    # Save or print
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format_for_encoder:
            output_path.write_text(output)
        else:
            with output_path.open("w") as f:
                json.dump(analysis, f, indent=2)

        logger.info(f"✓ Saved to: {args.output}")
    else:
        if args.format_for_encoder:
            print(output)
        else:
            print(json.dumps(analysis, indent=2))

    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
