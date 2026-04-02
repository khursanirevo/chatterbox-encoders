"""
Tests for Qwen3-Omni text analyzer.
"""

import numpy as np


def test_qwen_analyzer_initialization():
    """Test Qwen3-Omni analyzer can be initialized."""
    from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer

    analyzer = QwenOmniAnalyzer(device="cpu")

    assert analyzer is not None
    assert analyzer.device == "cpu"


def test_qwen_analyzer_extract_text_analysis():
    """Test Qwen3-Omni extracts text analysis from audio."""
    from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer

    analyzer = QwenOmniAnalyzer(device="cpu")

    # Create dummy audio (1 second at 16kHz)
    audio = np.random.randn(16000).astype(np.float32)

    # Extract text analysis
    analysis = analyzer.analyze_audio(audio)

    # Check structure
    assert isinstance(analysis, dict)
    assert "emotion" in analysis
    assert "profile" in analysis
    assert "mood" in analysis
    assert "speed" in analysis
    assert "prosody" in analysis
    assert "pitch_timbre" in analysis
    assert "style" in analysis
    assert "notes" in analysis
    assert "caption" in analysis

    # Check all values are non-empty strings
    for key, value in analysis.items():
        assert isinstance(value, str)
        assert len(value) > 0, f"{key} is empty"


def test_qwen_analyzer_format_for_encoder():
    """Test formatting text analysis for text encoder."""
    from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer

    analyzer = QwenOmniAnalyzer(device="cpu")

    # Create dummy analysis
    analysis = {
        "emotion": "happy",
        "profile": "young female speaker",
        "mood": "cheerful",
        "speed": "moderate",
        "prosody": "rising intonation",
        "pitch_timbre": "high-pitched, bright",
        "style": "conversational",
        "notes": "background noise present",
        "caption": "A cheerful greeting with warm tone",
    }

    # Format for encoder
    formatted = analyzer.format_for_encoder(analysis)

    # Check it's a string
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "emotion" in formatted.lower()
