# Chatterbox Encoders - Standalone Package

**Audio → Speech Tokens → Audio** reconstruction package.

Completely independent with HuggingFace weights. No LLM components.

## Quick Setup (5 minutes)

### Step 1: Clone and Install

```bash
git clone https://github.com/khursanirevo/chatterbox-encoders.git
cd chatterbox-encoders
pip install -e .
```

### Step 2: Download Weights

```bash
python download_weights.py
```

This downloads ~1.01 GB from HuggingFace:
- `s3gen.safetensors` (1007 MB)
- `ve.safetensors` (5.4 MB)
- `tokenizer.json` (25 KB)

### Step 3: Verify Installation

```bash
python setup_standalone.py
```

Expected output:
```
MAE:  0.0758
MSE:  0.0214
Threshold: < 0.1

MAE < 0.1:  ✅ PASS
MSE < 0.1:  ✅ PASS
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
✅ **Independent**: No external deps (weights from HF)

## Package Structure

```
chatterbox-encoders/
├── weights/              # Model weights (downloaded from HF)
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

## Reconstruction Quality

On 10-second audio clips:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **MAE** | 0.07-0.08 | < 0.1 | ✅ Pass |
| **MSE** | 0.02-0.03 | < 0.1 | ✅ Pass |

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
