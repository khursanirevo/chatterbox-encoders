"""
Text analysis for audio understanding.

This module provides text-based analysis of audio using Qwen3-Omni,
and learns to map text analysis to audio embeddings.
"""

from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer
from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

__all__ = [
    "QwenOmniAnalyzer",
    "TextToAudioEmbedding",
]
