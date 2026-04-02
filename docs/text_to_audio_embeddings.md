# Text-to-Audio-Embeddings Learning

## Overview

This document describes the text-to-audio-embeddings learning system, which enables text-only audio generation by learning to map rich text analysis to audio-compatible embeddings.

## Motivation

During inference, we want to generate audio from text descriptions only, without requiring reference audio. This is achieved by:

1. **Training Phase**: Learn to map text analysis → audio embeddings using reference audio
2. **Inference Phase**: Generate audio from text analysis alone

## Architecture

### Training Phase

```
Reference Audio → Qwen3-Omni → Text Analysis
                                        ↓
                                Text Encoder → 32×1024 tokens
                                        ↓
                                  Learn to match (MSE)
                                        ↓
Reference Audio → Voice Encoder → Perceiver → 32×1024 tokens (ground truth)
```

### Inference Phase

```
Text Analysis → Text Encoder → 32×1024 tokens → Audio Generation
```

## Components

### 1. QwenOmniAnalyzer

Extracts rich text analysis from audio using Qwen3-Omni.

**Output Fields:**
- `emotion`: Emotion of the speech
- `profile`: Speaker profile
- `mood`: Mood of the speech
- `speed`: Speaking speed
- `prosody`: Prosody, rhythm
- `pitch_timbre`: Pitch, voice quality
- `style`: Style of utterance
- `notes`: Other relevant notes
- `caption`: A comprehensive caption integrating all elements

**Usage:**
```python
from chatterbox_encoders.text_analysis import QwenOmniAnalyzer

analyzer = QwenOmniAnalyzer(device="cuda")
analysis = analyzer.analyze_from_file("speech.wav")
print(analysis['caption'])
# A cheerful greeting with warm tone
```

### 2. TextToAudioEmbedding

T5-based encoder that maps text analysis to 32×1024 audio tokens.

**Architecture:**
```
Text → T5 Encoder (512-dim) → Projection → 1024-dim → 32 tokens
```

**Usage:**
```python
from chatterbox_encoders.text_analysis import TextToAudioEmbedding

encoder = TextToAudioEmbedding(device="cuda")
text = "Emotion: happy\nCaption: A cheerful greeting"
embeddings = encoder(text)  # (1, 32, 1024)
```

## Training

### Data Preparation

Create a JSON file with paths to reference audio:

```json
[
    "data/audio/speech_001.wav",
    "data/audio/speech_002.wav",
    "data/audio/speech_003.wav"
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
1. Extract text analysis with Qwen3-Omni
2. Get ground truth tokens from voice encoder + Perceiver
3. Train text encoder to predict ground truth from text analysis
4. Minimize MSE loss between predicted and ground truth tokens

## Inference

### Text-Only Generation

After training, generate audio from text analysis alone:

```python
from chatterbox_encoders.text_analysis import QwenOmniAnalyzer, TextToAudioEmbedding

# Load trained text encoder
encoder = TextToAudioEmbedding(device="cuda")
encoder.load("checkpoints/text_encoder/best.pt")

# Provide text analysis (manually written or from Qwen3-Omni)
text_analysis = """
Emotion: happy
Profile: young female speaker
Mood: cheerful
Speed: moderate
Prosody: rising intonation
Pitch/Timbre: high-pitched, bright
Style: conversational
Notes: greeting
Caption: A cheerful greeting with warm tone
"""

# Encode to audio tokens
audio_tokens = encoder(text_analysis)  # (1, 32, 1024)

# Use audio_tokens for generation (with your audio generation model)
# generated_audio = audio_model.generate(audio_tokens)
```

## Utility Scripts

### Analyze Audio

Extract text analysis from audio files:

```bash
python scripts/analyze_audio_with_qwen.py \
    audio/speech.wav \
    --output analysis.json
```

Format for text encoder:

```bash
python scripts/analyze_audio_with_qwen.py \
    audio/speech.wav \
    --format-for-encoder \
    --output analysis.txt
```

## Model Details

### Qwen3-Omni Analyzer
- **Model**: Qwen/Qwen2-Audio-7B-Instruct
- **Purpose**: Audio-to-text analysis
- **Output**: 9 text fields (emotion, profile, mood, etc.)

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
- [ ] Add data augmentation for text analysis
- [ ] Experiment with larger T5 variants (base, large)
- [ ] Add cross-attention between learned queries and text embeddings
- [ ] Benchmark reconstruction quality vs learned embeddings
- [ ] Add fine-tuning of T5 encoder for better performance
