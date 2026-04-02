"""
Constants for Chatterbox encoders.

This module defines all constant values used across the encoding components.
"""

from typing import Final

# =============================================================================
# Sample Rates
# =============================================================================
S3_SR: Final[int] = 16_000  # S3Tokenizer input sample rate
S3GEN_SR: Final[int] = 24_000  # S3Gen decoder sample rate
VOICE_ENCODER_SR: Final[int] = 16_000  # Voice encoder sample rate

# =============================================================================
# S3Tokenizer Constants
# =============================================================================
S3_HOP: Final[int] = 160  # Hop size for STFT (100 frames/sec)
S3_TOKEN_HOP: Final[int] = 640  # Hop size for tokens (25 tokens/sec)
S3_TOKEN_RATE: Final[int] = 25  # Tokens per second
SPEECH_VOCAB_SIZE: Final[int] = 6_561  # Speech token vocabulary size
N_FFT_S3: Final[int] = 400  # FFT size for S3Tokenizer
N_MELS_S3: Final[int] = 128  # Number of mel bands for S3Tokenizer

# =============================================================================
# S3Gen Decoder Constants
# =============================================================================
N_FFT_24K: Final[int] = 1_920  # FFT size for 24kHz
HOP_SIZE_24K: Final[int] = 480  # Hop size for 24kHz
WIN_SIZE_24K: Final[int] = 1_920  # Window size for 24kHz
N_MELS_24K: Final[int] = 80  # Number of mel bands for 24kHz

# =============================================================================
# Voice Encoder Constants
# =============================================================================
N_MELS_VE: Final[int] = 40  # Number of mel bands for voice encoder
VE_HIDDEN_SIZE: Final[int] = 256  # LSTM hidden size
SPEAKER_EMBED_SIZE: Final[int] = 256  # Speaker embedding dimension
VE_PARTIAL_FRAMES: Final[int] = 96  # Partial utterance frames

# =============================================================================
# Model Dimensions
# =============================================================================
MODEL_DIM: Final[int] = 1_024  # Common model dimension (LLaMA hidden size)
GPT2_DIM: Final[int] = 1_024  # GPT-2 hidden size

# =============================================================================
# Perceiver Constants
# =============================================================================
PERCEIVER_QUERIES: Final[int] = 32  # Number of Perceiver query tokens
PERCEIVER_QUERY_DIM: Final[int] = 1_024  # Perceiver query dimension
PERCEIVER_NUM_HEADS: Final[int] = 4  # Number of attention heads

# =============================================================================
# Text Tokenizer Constants
# =============================================================================
START_TEXT_TOKEN: Final[int] = 255  # Start of text token
STOP_TEXT_TOKEN: Final[int] = 0  # End of text token
START_SPEECH_TOKEN: Final[int] = 6_561  # Start of speech token
STOP_SPEECH_TOKEN: Final[int] = 6_562  # End of speech token

ENGLISH_VOCAB_SIZE: Final[int] = 704  # English tokenizer vocabulary
MULTILINGUAL_VOCAB_SIZE: Final[int] = 2_454  # Multilingual tokenizer vocabulary

# =============================================================================
# Special Tokens
# =============================================================================
SOT: Final[str] = "[START]"
EOT: Final[str] = "[STOP]"
UNK: Final[str] = "[UNK]"
SPACE: Final[str] = "[SPACE]"
PAD: Final[str] = "[PAD]"

# =============================================================================
# Conditioning Lengths
# =============================================================================
ENC_COND_LEN_REGULAR: Final[int] = 6 * S3_SR  # 6 seconds @ 16kHz (regular)
ENC_COND_LEN_TURBO: Final[int] = 15 * S3_SR  # 15 seconds @ 16kHz (turbo)
DEC_COND_LEN: Final[int] = 10 * S3GEN_SR  # 10 seconds @ 24kHz

# =============================================================================
# Padding
# =============================================================================
S3_QUIET_PAD: Final[int] = 6_563  # Quiet padding token
S3_PAD_TOKEN: Final[int] = 6_562  # Padding token (same as stop)
