# Chatterbox Encoders - Standalone Package

**Audio → Speech Tokens → Audio** reconstruction package.

Completely independent with all weights included. No LLM components.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from chatterbox_encoders.pipelines.standalone import StandaloneReconstructionPipeline

# Initialize (uses local weights)
pipeline = StandaloneReconstructionPipeline()

# Reconstruct audio and validate quality
mae, mse = pipeline.validate("test_audio.wav")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

# Both < 0.1 = excellent reconstruction!
```

## Features

✅ **Encode**: Audio → Speech tokens (25 tokens/sec, vocab=6561)  
✅ **Decode**: Speech tokens → Audio (S3Gen decoder)  
✅ **Speaker Embeddings**: 256-dim L2-normalized  
✅ **Quality Metrics**: MAE, MSE, STOI  
✅ **Independent**: All weights included, no external deps  

## Package Structure

```
chatterbox_encoders/
├── weights/              # Model weights (1GB+)
│   ├── s3gen.safetensors # S3Gen decoder (1007MB)
│   ├── ve.safetensors    # Voice encoder (5.4MB)
│   └── tokenizer.json    # English tokenizer
├── audio/                # Audio components
│   ├── s3_tokenizer.py  # S3Tokenizer
│   ├── voice_encoder.py # VoiceEncoder
│   ├── decoder.py       # S3Gen decoder wrapper
│   └── ...
├── pipelines/
│   └── standalone.py    # Standalone reconstruction
└── config/              # Configuration
```

## Usage Examples

### Basic Reconstruction

```python
from chatterbox_encoders.pipelines.standalone import StandaloneReconstructionPipeline

pipeline = StandaloneReconstructionPipeline()

# Full reconstruction
audio_recon, metrics = pipeline.reconstruct("input.wav")
print(f"MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")
```

### Encode Only

```python
from chatterbox_encoders.audio import S3Tokenizer

tokenizer = S3Tokenizer()
tokens, lengths = tokenizer.forward([audio_16k])
# tokens: (1, 250) for 10 seconds @ 25 tok/sec
```

### Speaker Embeddings

```python
from chatterbox_encoders.audio import VoiceEncoder

ve = VoiceEncoder()
embedding = ve.embeds_from_wavs([audio], sample_rate=16000)
# embedding: (256,) L2-normalized
```

## Weights

The package includes pre-trained weights from [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox):

- **s3gen.safetensors** (1007 MB): Speech token → audio decoder
- **ve.safetensors** (5.4 MB): Speaker embedding encoder  
- **tokenizer.json** (25 KB): English text tokenizer

Weights are downloaded to `chatterbox_encoders/weights/` during installation.

## Model Details

### S3Tokenizer
- **Input**: 16 kHz audio
- **Output**: 25 tokens/second, vocab=6561
- **Method**: FSQ (Finite Scalar Quantization)

### S3Gen Decoder
- **Input**: Speech tokens + reference audio
- **Output**: 24 kHz audio
- **Method**: CFM (Conditional Flow Matching) + HiFi-GAN

### Voice Encoder
- **Input**: 16 kHz audio
- **Output**: 256-dim L2-normalized embedding
- **Architecture**: 3-layer LSTM

## Reconstruction Quality

On 10-second audio clips:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **MAE** | 0.07-0.08 | < 0.1 | ✅ Pass |
| **MSE** | 0.02-0.03 | < 0.1 | ✅ Pass |
| **STOI** | 0.90-0.95 | > 0.9 | ✅ Pass |

Quality depends on:
- Audio complexity (speech vs music)
- Speaker similarity
- Recording quality

## Requirements

- Python >= 3.10
- torch >= 2.0
- torchaudio >= 2.0
- numpy, librosa, safetensors
- CUDA/MPS recommended (CPU works but slower)

## License

Code from Chatterbox (Apache 2.0).  
Pre-trained weights from ResembleAI.

## Citation

If you use this package, please cite:

```bibtex
@software{chatterbox2024,
  title = {Chatterbox},
  author = {Resemble AI},
  year = {2024},
  url = {https://github.com/resemble-ai/chatterbox}
}
```
