"""
Plot mel spectrograms for reconstruction analysis (Jupyter Notebook version).

Usage in Jupyter:
    import matplotlib.pyplot as plt
    from plot_reconstruction_notebook import plot_reconstruction_comparison

    # Load your audio files
    import librosa
    audio_orig, sr = librosa.load("original.wav", sr=24000)
    audio_recon, _ = librosa.load("reconstructed.wav", sr=24000)

    # Plot
    fig = plot_reconstruction_comparison(audio_orig, audio_recon, sr)
    plt.show()
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display


def compute_mel_spec(audio, sr=24000, n_mels=128, n_fft=2048, hop_length=512):
    """Compute mel spectrogram in dB."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def compute_normalized_energy_difference(mel_orig, mel_recon):
    """Compute normalized energy difference."""
    diff = np.abs(mel_orig - mel_recon)
    energy_range = np.max(mel_orig) - np.min(mel_orig)
    if energy_range > 0:
        diff_normalized = diff / energy_range
    else:
        diff_normalized = diff
    return diff_normalized


def plot_reconstruction_comparison(
    audio_orig,
    audio_recon,
    sr=24000,
    figsize=(14, 12)
):
    """
    Plot side-by-side mel spectrogram comparison.

    Args:
        audio_orig: Original audio waveform
        audio_recon: Reconstructed audio waveform
        sr: Sample rate
        figsize: Figure size

    Returns:
        matplotlib figure
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

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot 1: Original
    img1 = librosa.display.specshow(
        mel_orig,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Original Audio', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mel Freq')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # Plot 2: Reconstructed
    img2 = librosa.display.specshow(
        mel_recon,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        ax=axes[1],
        cmap='viridis'
    )
    axes[1].set_title('Reconstructed Audio', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Mel Freq')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    # Plot 3: Difference
    img3 = axes[2].imshow(
        diff_norm,
        aspect='auto',
        origin='lower',
        cmap='inferno',
        extent=[0, len(audio_orig) / sr, 0, 128]
    )
    axes[2].set_title('Normalized Energy Difference', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Mel Freq')
    fig.colorbar(img3, ax=axes[2])

    plt.tight_layout()

    return fig


def show_audio_comparison(audio_orig, audio_recon, sr=24000):
    """Display audio players for comparison."""
    print("🎵 Original Audio:")
    display(Audio(audio_orig, rate=sr))

    print("\n🔊 Reconstructed Audio:")
    display(Audio(audio_recon, rate=sr))


def analyze_reconstruction(audio_orig, audio_recon, sr=24000):
    """
    Complete reconstruction analysis with plots and audio playback.

    Args:
        audio_orig: Original audio waveform
        audio_recon: Reconstructed audio waveform
        sr: Sample rate
    """
    # Trim to same length
    min_len = min(len(audio_orig), len(audio_recon))
    audio_orig = audio_orig[:min_len]
    audio_recon = audio_recon[:min_len]

    # Calculate metrics
    mae = np.mean(np.abs(audio_orig - audio_recon))
    mse = np.mean((audio_orig - audio_recon) ** 2)

    # Mel spectrograms
    mel_orig = compute_mel_spec(audio_orig, sr)
    mel_recon = compute_mel_spec(audio_recon, sr)
    diff_norm = compute_normalized_energy_difference(mel_orig, mel_recon)

    # Print statistics
    print("=" * 60)
    print("RECONSTRUCTION ANALYSIS")
    print("=" * 60)
    print(f"Duration: {len(audio_orig)/sr:.2f} seconds")
    print(f"\nWaveform Metrics:")
    print(f"  MAE:  {mae:.4f} {'✅' if mae < 0.1 else '❌'} (threshold: < 0.1)")
    print(f"  MSE:  {mse:.4f} {'✅' if mse < 0.1 else '❌'} (threshold: < 0.1)")
    print(f"\nSpectrogram Difference:")
    print(f"  Mean: {np.mean(diff_norm):.4f}")
    print(f"  Max:  {np.max(diff_norm):.4f}")
    print("=" * 60)

    # Show plot
    fig = plot_reconstruction_comparison(audio_orig, audio_recon, sr)
    plt.show()

    # Show audio players
    show_audio_comparison(audio_orig, audio_recon, sr)

    return {
        "mae": mae,
        "mse": mse,
        "spec_diff_mean": np.mean(diff_norm),
        "spec_diff_max": np.max(diff_norm),
        "figure": fig
    }


# Example usage in Jupyter:
if __name__ == "__main__":
    import librosa

    # Load audio files
    audio_orig, sr = librosa.load("test_audio.wav", sr=24000)
    audio_recon, _ = librosa.load("test_audio_manual_reconstructed.wav", sr=24000)

    # Analyze
    results = analyze_reconstruction(audio_orig, audio_recon, sr)
