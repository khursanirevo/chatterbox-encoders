"""
Configuration and constants for Chatterbox encoders.

This module provides default parameters and constants used across
the encoding components.
"""

from chatterbox_encoders.config.defaults import *
from chatterbox_encoders.config.constants import *

__all__ = [
    # Defaults
    "DEFAULT_SAMPLE_RATE_16K",
    "DEFAULT_SAMPLE_RATE_24K",
    "DEFAULT_S3_TOKEN_RATE",
    "DEFAULT_S3_VOCAB_SIZE",
    "DEFAULT_SPEAKER_EMBED_SIZE",
    "DEFAULT_MODEL_DIM",
    "DEFAULT_PERCEIVER_QUERIES",
    # Constants
    "S3_SR",
    "S3GEN_SR",
    "S3_TOKEN_RATE",
    "S3_VOCAB_SIZE",
    "SPEAKER_EMBED_SIZE",
    "MODEL_DIM",
    "N_FFT_16K",
    "N_FFT_24K",
]
