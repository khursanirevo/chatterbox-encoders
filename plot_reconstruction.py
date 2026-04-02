"""
Plot mel spectrograms for original, reconstructed audio and their difference.

This script visualizes:
1. Original audio mel spectrogram
2. Reconstructed audio mel spectrogram
3. Normalized energy difference
"""

import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path


def compute_mel_spec(audio, sr=24000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Compute mel spectrogram.

    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Mel spectrogram in dB
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def compute_normalized_energy_difference(mel_orig, mel_recon):
    """
    Compute normalized energy difference between two mel spectrograms.

    Args:
        mel_orig: Original mel spectrogram (dB)
        mel_recon: Reconstructed mel spectrogram (dB)

    Returns:
        Normalized energy difference
    """
    # Compute absolute difference
    diff = np.abs(mel_orig - mel_recon)

    # Normalize by max energy in original
    # Use the range of original spectrogram for normalization
    energy_range = np.max(mel_orig) - np.min(mel_orig)
    if energy_range > 0:
        diff_normalized = diff / energy_range
    else:
        diff_normalized = diff

    return diff_normalized


def plot_comparison(
    audio_orig,
    audio_recon,
    sr=24000,
    save_path="reconstruction_comparison.png",
    title="Audio Reconstruction Comparison"
):
    """
    Plot side-by-side comparison of mel spectrograms.

    Args:
        audio_orig: Original audio waveform
        audio_recon: Reconstructed audio waveform
        sr: Sample rate
        save_path: Path to save the plot
        title: Figure title
    """
    # Trim to same length
    min_len = min(len(audio_orig), len(audio_recon))
    audio_orig = audio_orig[:min_len]
    audio_recon = audio_recon[:min_len]

    # Compute mel spectrograms
    mel_orig = compute_mel_spec(audio_orig, sr)
    mel_recon = compute_mel_spec(audio_recon, sr)

    # Compute normalized difference
    diff_norm = compute_normalized_energy_difference(mel_orig, mel_recon)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Original mel spectrogram
    img1 = librosa.display.specshow(
        mel_orig,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Original Audio Mel Spectrogram', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mel Frequency', fontsize=12)
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB', label='Power (dB)')

    # Plot 2: Reconstructed mel spectrogram
    img2 = librosa.display.specshow(
        mel_recon,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=axes[1],
        cmap='viridis'
    )
    axes[1].set_title('Reconstructed Audio Mel Spectrogram', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Mel Frequency', fontsize=12)
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB', label='Power (dB)')

    # Plot 3: Normalized energy difference
    img3 = axes[2].imshow(
        diff_norm,
        aspect='auto',
        origin='lower',
        cmap='inferno',
        extent=[0, len(audio_orig) / sr, 0, 128]
    )
    axes[2].set_title('Normalized Energy Difference', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Mel Frequency', fontsize=12)
    fig.colorbar(img3, ax=axes[2], label='Normalized Difference')

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # Add statistics
    mae = np.mean(np.abs(audio_orig - audio_recon))
    mse = np.mean((audio_orig - audio_recon) ** 2)
    diff_mean = np.mean(diff_norm)
    diff_max = np.max(diff_norm)

    stats_text = (
        f"Waveform Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}\n"
        f"Spectrogram Difference - Mean: {diff_mean:.4f}, Max: {diff_max:.4f}"
    )
    fig.text(0.5, 0.005, stats_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {save_path}")

    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"   Waveform MAE:  {mae:.4f}")
    print(f"   Waveform MSE:  {mse:.4f}")
    print(f"   Spectrogram Difference Mean: {diff_mean:.4f}")
    print(f"   Spectrogram Difference Max:  {diff_max:.4f}")

    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot mel spectrogram comparison for reconstructed audio"
    )
    parser.add_argument(
        "original",
        help="Path to original audio file"
    )
    parser.add_argument(
        "reconstructed",
        help="Path to reconstructed audio file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="reconstruction_comparison.png",
        help="Output path for the plot (default: reconstruction_comparison.png)"
    )
    parser.add_argument(
        "--title",
        "-t",
        default="Audio Reconstruction Comparison",
        help="Figure title"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=24000,
        help="Sample rate (default: 24000)"
    )

    args = parser.parse_args()

    # Load audio files
    print(f"Loading original audio: {args.original}")
    audio_orig, sr_orig = librosa.load(args.original, sr=args.sr, mono=True)

    print(f"Loading reconstructed audio: {args.reconstructed}")
    audio_recon, sr_recon = librosa.load(args.reconstructed, sr=args.sr, mono=True)

    print(f"Original duration: {len(audio_orig)/sr_orig:.2f}s")
    print(f"Reconstructed duration: {len(audio_recon)/sr_recon:.2f}s")

    # Plot comparison
    plot_comparison(
        audio_orig,
        audio_recon,
        sr=args.sr,
        save_path=args.output,
        title=args.title
    )

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
