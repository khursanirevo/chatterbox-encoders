"""
Tests for Qwen3-Omni audio encoder.
"""

import numpy as np
import torch


def test_qwen_omni_encoder_initialization():
    """Test Qwen3-Omni encoder can be initialized."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder

    encoder = QwenOmniAudioEncoder(device="cpu")

    assert encoder is not None
    assert encoder.output_dim == 1280
    assert encoder.device == "cpu"


def test_qwen_omni_encoder_forward():
    """Test Qwen3-Omni encoder processes audio correctly."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder

    encoder = QwenOmniAudioEncoder(device="cpu")

    # Create dummy audio (1 second at 16kHz)
    audio = np.random.randn(16000).astype(np.float32)

    # Encode
    with torch.no_grad():
        embeddings = encoder.encode_audio(audio)

    # Check output shape: (seq_len, 1280)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 1280
    assert embeddings.shape[0] > 0  # Should have some sequence length


def test_qwen_omni_encoder_batch():
    """Test Qwen3-Omni encoder handles batch audio."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder

    encoder = QwenOmniAudioEncoder(device="cpu")

    # Create dummy audio batch (2 samples, 1 second each)
    audio_batch = [np.random.randn(16000).astype(np.float32) for _ in range(2)]

    # Encode batch
    with torch.no_grad():
        embeddings = encoder.encode_audio_batch(audio_batch)

    # Check output shape: (batch, seq_len, 1280)
    assert embeddings.ndim == 3
    assert embeddings.shape[0] == 2
    assert embeddings.shape[2] == 1280


def test_qwen_projector():
    """Test Qwen3-Omni projection layer."""
    from chatterbox_encoders.audio.qwen_projector import QwenProjector

    projector = QwenProjector(device="cpu")

    # Create dummy Qwen embeddings (batch, seq_len, 1280)
    qwen_emb = torch.randn(1, 100, 1280)

    # Project
    projected = projector(qwen_emb)

    # Check output shape: (batch, seq_len, 1024)
    assert projected.shape == (1, 100, 1024)


def test_qwen_projector_with_perceiver():
    """Test full pipeline: Qwen -> Projector -> Perceiver."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder
    from chatterbox_encoders.audio.qwen_projector import QwenProjector

    from chatterbox_encoders.audio.perceiver import PerceiverResampler

    # Initialize components
    encoder = QwenOmniAudioEncoder(device="cpu")
    projector = QwenProjector(device="cpu")
    perceiver = PerceiverResampler(num_queries=32, query_dim=1024, embedding_dim=1024)
    perceiver = perceiver.to("cpu")

    # Create dummy audio
    audio = np.random.randn(16000).astype(np.float32)

    # Full pipeline
    with torch.no_grad():
        # Encode with Qwen
        qwen_emb = encoder.encode_audio(audio)  # (seq_len, 1280)
        qwen_emb = qwen_emb.unsqueeze(0)  # (1, seq_len, 1280)

        # Project to 1024
        projected = projector(qwen_emb)  # (1, seq_len, 1024)

        # Compress with Perceiver
        compressed = perceiver(projected)  # (1, 32, 1024)

    # Check final output
    assert compressed.shape == (1, 32, 1024)
