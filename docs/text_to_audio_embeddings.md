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

Sentence Transformer-based encoder that maps user-provided text labels to 32×1024 audio tokens.

**Architecture:**
```
Text → Sentence Transformer (768-dim) → Projection → 1024-dim → 32 tokens
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

**Step 1: Organize your audio files**

Create a directory with your audio files:

```bash
mkdir -p data/train_audio
mkdir -p data/val_audio

# Copy your audio files to these directories
# cp /path/to/train/*.wav data/train_audio/
# cp /path/to/val/*.wav data/val_audio/
```

**Step 2: Create labels.json**

Use the helper script to create a labels template:

```bash
python scripts/prepare_training_data.py --data-dir data/train_audio
python scripts/prepare_training_data.py --data-dir data/val_audio
```

This creates `labels.json` in each directory:

```json
{
  "audio_001.wav": "",
  "audio_002.wav": "",
  "audio_003.wav": ""
}
```

**Step 3: Fill in text labels**

Edit each `labels.json` and add your text labels:

```json
{
  "audio_001.wav": "A cheerful greeting with warm tone and friendly delivery",
  "audio_002.wav": "Sad farewell with quiet voice and slow tempo",
  "audio_003.wav": "Excited announcement with energetic delivery and fast pace"
}
```

**Expected directory structure:**

```
data/train_audio/
    ├── audio_001.wav
    ├── audio_002.wav
    ├── audio_003.wav
    └── labels.json

data/val_audio/
    ├── audio_001.wav
    ├── audio_002.wav
    └── labels.json
```

### Training Command

```bash
python scripts/train_text_encoder.py \
    --data-dir data/train_audio \
    --val-data-dir data/val_audio \
    --output-dir checkpoints/text_encoder \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda
```

**Full options:**

```bash
python scripts/train_text_encoder.py \
    --data-dir DATA_DIR \
    --val-data-dir VAL_DATA_DIR \
    --output-dir checkpoints/text_encoder \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --device auto \
    --seed 42 \
    --checkpoint checkpoints/text_encoder/best.pt \
    --max-duration 30.0
```

### Training Loop

For each (audio, text_label) pair in the dataset:
1. **Extract ground truth**: Audio → S3Tokenizer → SpeechTokenEmbedding → Perceiver → 32×1024 tokens
2. **Get prediction**: Text label → Sentence Transformer → Projection → 32×1024 tokens
3. **Compute loss**: MSE(prediction, ground_truth)
4. **Update weights**: Backpropagation through projection + query embeddings (Sentence Transformer is frozen)

### Training Output

The script will:
- Save checkpoints to `--output-dir` after each epoch
- Save best model based on validation loss
- Log training progress with tqdm progress bars
- Display train/validation loss after each epoch

**Output files:**
- `epoch_1.pt`, `epoch_2.pt`, ... - Checkpoints after each epoch
- `best.pt` - Best model based on validation loss

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
- **Base**: sentence-transformers/all-mpnet-base-v2 (~420MB)
- **Output dim**: 1024 (for Perceiver compatibility)
- **Num queries**: 32 (for Perceiver compatibility)
- **Trainable params**: Projection layer + query embeddings (~2M params)
- **Frozen params**: Sentence Transformer (reduces training cost)

## Performance Considerations

- **Training speed**: ~10-100 examples/second (GPU dependent)
- **Inference speed**: Real-time (Sentence Transformer is fast)
- **Memory**: ~2GB GPU memory for training
- **Model size**: ~430MB (Sentence Transformer + projection)

## Future Work

- [ ] Implement complete AudioTextDataset with voice encoder integration
- [ ] Add data augmentation for text labels
- [ ] Experiment with different Sentence Transformer models
- [ ] Add cross-attention between learned queries and text embeddings
- [ ] Benchmark reconstruction quality vs learned embeddings
- [ ] Add fine-tuning of Sentence Transformer for better performance
