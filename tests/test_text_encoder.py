"""
Tests for text-to-audio-embedding encoder.
"""

import tempfile
from pathlib import Path

import torch


def test_text_encoder_initialization():
    """Test text encoder can be initialized."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")

    assert encoder is not None
    assert encoder.output_dim == 1024
    assert encoder.num_queries == 32
    assert encoder.device == "cpu"


def test_text_encoder_forward():
    """Test text encoder produces correct output shape."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")

    # Create sample text analysis
    text_analysis = (
        "Emotion: happy\n"
        "Profile: young female speaker\n"
        "Mood: cheerful\n"
        "Speed: moderate\n"
        "Prosody: rising intonation\n"
        "Pitch/Timbre: high-pitched, bright\n"
        "Style: conversational\n"
        "Notes: greeting\n"
        "Caption: A cheerful greeting with warm tone"
    )

    # Encode
    with torch.no_grad():
        embeddings = encoder(text_analysis)

    # Check output shape: (1, 32, 1024)
    assert embeddings.ndim == 3
    assert embeddings.shape[0] == 1  # batch
    assert embeddings.shape[1] == 32  # num queries
    assert embeddings.shape[2] == 1024  # embedding dim


def test_text_encoder_batch():
    """Test text encoder handles batch input."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")

    # Create batch of text analyses
    text_batch = [
        "Emotion: happy\nCaption: A cheerful greeting",
        "Emotion: sad\nCaption: A somber statement",
    ]

    # Encode batch
    with torch.no_grad():
        embeddings = encoder(text_batch)

    # Check output shape: (2, 32, 1024)
    assert embeddings.shape == (2, 32, 1024)


def test_text_encoder_training():
    """Test text encoder can be trained."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    # Sample inputs
    text = "Emotion: happy\nCaption: A cheerful greeting"
    target = torch.randn(1, 32, 1024)  # Target from voice encoder

    # Training step
    optimizer.zero_grad()
    prediction = encoder(text)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()

    # Check loss decreased
    assert loss.item() > 0


def test_text_encoder_save_load():
    """Test text encoder can be saved and loaded."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")
    text = "Emotion: happy\nCaption: A cheerful greeting"

    # Get output before saving
    with torch.no_grad():
        output_before = encoder(text)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "text_encoder.pt"
        encoder.save(checkpoint_path)

        # Load new instance
        encoder2 = TextToAudioEmbedding(device="cpu")
        encoder2.load(checkpoint_path)

        # Get output after loading
        with torch.no_grad():
            output_after = encoder2(text)

        # Check outputs are the same
        assert torch.allclose(output_before, output_after, atol=1e-6)
