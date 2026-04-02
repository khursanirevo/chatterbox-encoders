"""
Chatterbox Encoders - Audio and Text Encoding Components

This package provides encoding components from the Chatterbox TTS system
for extracting speech tokens, speaker embeddings, and audio features.

Main Components:
- S3Tokenizer: Audio → Speech tokens (25 tokens/sec)
- VoiceEncoder: Audio → 256-dim speaker embedding
- PerceiverResampler: Variable-length → 32 tokens compression
- SpeakerProjector: 256-dim → 1024-dim projection
- EmotionProjector: Scalar → 1024-dim projection

Example Usage:
    >>> from chatterbox_encoders import S3Tokenizer, VoiceEncoder
    >>>
    >>> # Tokenize audio
    >>> tokenizer = S3Tokenizer()
    >>> tokens, lengths = tokenizer.forward([audio_16k])
    >>>
    >>> # Extract speaker embedding
    >>> ve = VoiceEncoder()
    >>> embedding = ve.embeds_from_wavs([audio], sample_rate=16000, as_spk=True)
"""

__version__ = "0.1.0"
__author__ = "Chatterbox Encoders Contributors"

# Core audio components
from chatterbox_encoders.audio.s3_tokenizer import S3Tokenizer
from chatterbox_encoders.audio.voice_encoder import VoiceEncoder
from chatterbox_encoders.audio.speaker_projector import SpeakerProjector
from chatterbox_encoders.audio.perceiver import PerceiverResampler
from chatterbox_encoders.audio.emotion import EmotionProjector

# Text components
from chatterbox_encoders.text.tokenizer_wrapper import text_to_tokens
from chatterbox_encoders.text.english_tokenizer import EnTokenizer as EnglishTokenizer
from chatterbox_encoders.text.normalizer import punc_norm

__all__ = [
    # Audio
    "S3Tokenizer",
    "VoiceEncoder",
    "SpeakerProjector",
    "PerceiverResampler",
    "EmotionProjector",
    # Text
    "text_to_tokens",
    "EnglishTokenizer",
    "punc_norm",
]
