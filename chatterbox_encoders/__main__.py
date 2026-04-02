#!/usr/bin/env python3
"""
Chatterbox Encoders - Main entry point

Usage:
    python -m chatterbox_encoders
    python -m chatterbox_encoders <audio_file>
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Chatterbox Encoders - Audio Reconstruction")
        print("=" * 50)
        print("\nUsage:")
        print("  python -m chatterbox_encoders <audio_file>")
        print("\nOr use the CLI:")
        print("  python -m chatterbox_encoders.pipelines.standalone <audio_file>")
        return 0
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        return 1
    
    from chatterbox_encoders.pipelines.standalone import StandaloneReconstructionPipeline
    
    print(f"Reconstructing: {audio_file}")
    print("-" * 50)
    
    pipeline = StandaloneReconstructionPipeline()
    audio_recon, metrics = pipeline.reconstruct(audio_file)
    
    print(f"\nResults:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  STOI: {metrics['stoi']:.4f}")
    
    if metrics['mae'] < 0.1:
        print(f"  ✅ Quality: EXCELLENT")
    else:
        print(f"  ⚠️  Quality: ACCEPTABLE")
    
    print(f"\nOutput saved to: {Path(audio_file).stem}_reconstructed.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
