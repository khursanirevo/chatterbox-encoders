"""
Audio encoding components for Chatterbox.

This module provides components for extracting features from audio:
- Speech tokenization (S3Tokenizer)
- Speaker embedding extraction (VoiceEncoder)
- Embedding projections (SpeakerProjector, EmotionProjector)
- Token compression (PerceiverResampler)
- Mel spectrogram extraction
"""

from chatterbox_encoders.audio.s3_tokenizer import S3Tokenizer
from chatterbox_encoders.audio.voice_encoder import VoiceEncoder
from chatterbox_encoders.audio.speaker_projector import SpeakerProjector
from chatterbox_encoders.audio.perceiver import PerceiverResampler
from chatterbox_encoders.audio.emotion import EmotionProjector
from chatterbox_encoders.audio.mel_extractor import (
    mel_spectrogram_16k,
    mel_spectrogram_24k,
    mel_spectrogram_40k,
)

__all__ = [
    "S3Tokenizer",
    "VoiceEncoder",
    "SpeakerProjector",
    "PerceiverResampler",
    "EmotionProjector",
    "mel_spectrogram_16k",
    "mel_spectrogram_24k",
    "mel_spectrogram_40k",
]
