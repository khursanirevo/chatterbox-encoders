"""
Default configuration parameters for Chatterbox encoders.

This module defines default values for all configurable parameters.
"""

from chatterbox_encoders.config.constants import (
    MODEL_DIM,
    S3_SR,
    S3GEN_SR,
    SPEAKER_EMBED_SIZE,
    PERCEIVER_QUERIES,
    PERCEIVER_QUERY_DIM,
    PERCEIVER_NUM_HEADS,
)

# =============================================================================
# Device Configuration
# =============================================================================
DEFAULT_DEVICE: str = "cuda"  # Default device for models
DEVICE_AUTO: str = "auto"  # Auto-detect device

# =============================================================================
# S3Tokenizer Defaults
# =============================================================================
S3TOKENIZER_NAME: str = "speech_tokenizer_v2_25hz"
S3TOKENIZER_NUM_MELS: int = 128
S3TOKENIZER_N_FFT: int = 400
S3TOKENIZER_HOP: int = 160

# =============================================================================
# Voice Encoder Defaults
# =============================================================================
VOICE_ENCODER_NUM_MELS: int = 40
VOICE_ENCODER_SAMPLE_RATE: int = 16000
VOICE_ENCODER_HIDDEN_SIZE: int = 256
VOICE_ENCODER_NUM_LAYERS: int = 3
VOICE_ENCODER_EMBED_SIZE: int = 256
VOICE_ENCODER_NORMALIZED_MELS: bool = False  # Use log-scale mels

# =============================================================================
# Perceiver Defaults
# =============================================================================
PERCEIVER_NUM_QUERIES: int = 32
PERCEIVER_QUERY_DIM: int = 1024
PERCEIVER_EMBEDDING_DIM: int = 1024
PERCEIVER_NUM_HEADS: int = 4
PERCEIVER_DROPOUT: float = 0.2

# =============================================================================
# Mel Spectrogram Defaults
# =============================================================================
MEL_16K_NUM_MELS: int = 128
MEL_16K_SAMPLE_RATE: int = 16000
MEL_16K_N_FFT: int = 400
MEL_16K_HOP: int = 160

MEL_24K_NUM_MELS: int = 80
MEL_24K_SAMPLE_RATE: int = 24000
MEL_24K_N_FFT: int = 1920
MEL_24K_HOP: int = 480
MEL_24K_WIN: int = 1920
MEL_24K_FMIN: int = 0
MEL_24K_FMAX: int = 8000

MEL_40K_NUM_MELS: int = 40
MEL_40K_SAMPLE_RATE: int = 16000

# =============================================================================
# Projection Defaults
# =============================================================================
SPEAKER_PROJ_INPUT_DIM: int = 256
SPEAKER_PROJ_OUTPUT_DIM: int = MODEL_DIM  # 1024

EMOTION_PROJ_INPUT_DIM: int = 1
EMOTION_PROJ_OUTPUT_DIM: int = MODEL_DIM  # 1024

# =============================================================================
# Text Processing Defaults
# =============================================================================
TEXT_NORMALIZE_LOWERCASE: bool = True
TEXT_NORMALIZE_NFKD: bool = True

# =============================================================================
# Reconstruction Defaults
# =============================================================================
RECONSTRUCTION_MAX_TOKENS: int = 1000
RECONSTRUCTION_TEMPERATURE: float = 0.8
RECONSTRUCTION_TOP_P: float = 0.95
RECONSTRUCTION_TOP_K: int = 1000
RECONSTRUCTION_REPETITION_PENALTY: float = 1.2

# =============================================================================
# Validation Thresholds
# =============================================================================
MAX_MAE_THRESHOLD: float = 0.1  # Maximum acceptable MAE
MAX_MSE_THRESHOLD: float = 0.1  # Maximum acceptable MSE
