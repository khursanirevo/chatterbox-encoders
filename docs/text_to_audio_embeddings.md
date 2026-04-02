# Text-to-Audio-Embeddings Learning

## Overview

This document describes the text-to-audio-embeddings learning system, which enables text-only audio generation by learning to map user-provided text labels to audio-compatible embeddings.

## Motivation

During inference, we want to generate audio from text descriptions only, without requiring reference audio. This is achieved by:

1. **Training Phase**: Learn to map text labels → audio embeddings using reference audio
2. **Inference Phase**: Generate audio from text labels alone

## Architecture

### Training Phase

```
Reference Audio → Text Labels (user-provided)
                                        ↓
                                Text Encoder → 32×1024 tokens
                                        ↓
                                  Learn to match (MSE)
                                        ↓
Reference Audio → Voice Encoder → Perceiver → 32×1024 tokens (ground truth)
```

### Inference Phase

```
Text Labels → Text Encoder → 32×1024 tokens → Audio Generation
```

## Components

### TextToAudioEmbedding

T5-based encoder that maps user-provided text labels to 32×1024 audio tokens.

**Architecture:**
```
Text → T5 Encoder (512-dim) → Projection → 1024-dim → 32 tokens
```

**Usage:**
```python
from chatterbox_encoders.text_analysis import TextToAudioEmbedding

encoder = TextToAudioEmbedding(device="cuda")
text = "A cheerful greeting with warm tone"
embeddings = encoder(text)  # (1, 32, 1024)
```

## Training

### Data Preparation

Create a JSON file mapping reference audio to text labels:

```json
[
    {
        "audio": "data/audio/speech_001.wav",
        "text": "A cheerful greeting with warm tone"
    },
    {
        "audio": "data/audio/speech_002.wav",
        "text": "Sad farewell with quiet voice"
    },
    {
        "audio": "data/audio/speech_003.wav",
        "text": "Excited announcement with energetic delivery"
    }
]
```

### Training Command

```bash
python scripts/train_text_encoder.py \
    --train-data data/train_audio.json \
    --val-data data/val_audio.json \
    --voice-envelope checkpoints/voice_encoder.pt \
    --output-dir checkpoints/text_encoder \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4
```

### Training Loop

For each audio in the dataset:
1. Get text label from user
2. Get ground truth tokens from voice encoder + Perceiver
3. Train text encoder to predict ground truth from text label
4. Minimize MSE loss between predicted and ground truth tokens

## Inference

### Text-Only Generation

After training, generate audio from text analysis alone:

```python
from chatterbox_encoders.text_analysis import TextToAudioEmbedding

# Load trained text encoder
encoder = TextToAudioEmbedding(device="cuda")
encoder.load("checkpoints/text_encoder/best.pt")

# Provide text label (describing desired audio characteristics)
text_label = "A cheerful greeting with warm tone"

# Encode to audio tokens
audio_tokens = encoder(text_label)  # (1, 32, 1024)

# Use audio_tokens for generation (with your audio generation model)
# generated_audio = audio_model.generate(audio_tokens)
```

## Model Details

### Text Encoder
- **Base**: T5-small (~240MB)
- **Output dim**: 1024 (for Perceiver compatibility)
- **Num queries**: 32 (for Perceiver compatibility)
- **Trainable params**: Projection layer + query embeddings (~2M params)
- **Frozen params**: T5 encoder (reduces training cost)

## Performance Considerations

- **Training speed**: ~10-100 examples/second (GPU dependent)
- **Inference speed**: Real-time (T5-small is fast)
- **Memory**: ~2GB GPU memory for training
- **Model size**: ~250MB (T5-small + projection)

## Future Work

- [ ] Implement complete AudioTextDataset with voice encoder integration
- [ ] Add data augmentation for text labels
- [ ] Experiment with larger T5 variants (base, large)
- [ ] Add cross-attention between learned queries and text embeddings
- [ ] Benchmark reconstruction quality vs learned embeddings
- [ ] Add fine-tuning of T5 encoder for better performance
