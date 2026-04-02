"""
Text analysis for audio understanding.

This module provides text-to-audio embedding mapping,
learning to map user-provided text labels to audio-compatible embeddings.
"""

from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

__all__ = [
    "TextToAudioEmbedding",
]
