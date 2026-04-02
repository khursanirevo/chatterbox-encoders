# Complete Chatterbox LLM Input Preparation Guide

## Overview

This guide explains how to prepare ALL inputs needed for Chatterbox LLM text-to-speech generation manually.

## Components Overview

The Chatterbox LLM requires the following inputs:

| # | Component | Shape | Purpose |
|---|-----------|-------|---------|
| 1 | **Text Tokens** | (1, seq_len) | Tokenized input text |
| 2 | **Speech Tokens** | (1, num_tokens) | Raw speech token IDs from audio |
| 3 | **Speech Embeddings** | (1, num_tokens, 1024) | Learned embeddings of speech tokens |
| 4 | **Compressed Speech** | (1, 32, 1024) | **Perceiver Resampler output** |
| 5 | **Speaker Embedding** | (1, 256) | Voice characteristics |
| 6 | **Speaker Projected** | (1, 1024) | Projected for LLM conditioning |

## Pipeline Flow

```
Input Text → Text Tokens → LLM Input
    ↓
Reference Audio → S3Tokenizer → Speech Tokens (371 tokens)
                                    ↓
                            Speech Embedding (371 × 1024)
                                    ↓
                            Perceiver Resampler → 32 Compressed Tokens
                                                              ↓
Reference Audio → VoiceEncoder → Speaker Embedding (256)
                                                      ↓
                                          SpeakerProjector → 1024 Projected
```

## Key Component: Perceiver Resampler

**Purpose:** Compress variable-length speech token embeddings to fixed 32 tokens

**Why?**
- Speech tokens vary by duration (e.g., 10s = 250 tokens, 14.84s = 371 tokens)
- LLM needs fixed-size context
- Perceiver uses cross-attention to compress while preserving information

**Transformation:**
```
Input:  (1, 371, 1024)  # Variable length embeddings
Output: (1, 32, 1024)   # Fixed 32 tokens
```

**Compression Ratio:** ~11.6x for 10s audio (371 → 32 tokens)

## Usage

### Basic Usage

```bash
python prepare_llm_inputs_with_perceiver.py \
  --text "Hello, world!" \
  --audio reference.wav \
  --output llm_inputs.pt
```

### Complete Example

```python
from prepare_llm_inputs_with_perceiver import CompleteLLMInputPreparer

# Initialize
preparer = CompleteLLMInputPreparer(device="cuda")

# Prepare all inputs
inputs = preparer.prepare_all_inputs(
    text="The quick brown fox jumps over the lazy dog.",
    audio_path="reference_speaker.wav"
)

# Access components
text_tokens = inputs["text"]["tokens"]              # (1, seq_len)
speech_tokens = inputs["speech"]["tokens"]          # (1, num_tokens)
speech_embeddings = inputs["speech"]["embeddings"]  # (1, num_tokens, 1024)
speech_compressed = inputs["speech"]["compressed"]  # (1, 32, 1024) ← KEY!
speaker_embedding = inputs["speaker"]["embedding"]  # (1, 256)
speaker_projected = inputs["speaker"]["projected"]  # (1, 1024)
```

### Without Perceiver Resampler

```bash
python prepare_llm_inputs_with_perceiver.py \
  --text "Hello" \
  --audio reference.wav \
  --no-perceiver \
  --output llm_inputs_no_perceiver.pt
```

## Output Comparison

### With Perceiver Resampler (Recommended)

```
Text tokens:        torch.Size([1, 31])
Speech tokens:      torch.Size([1, 371])
Speech embeddings:  torch.Size([1, 371, 1024])
Speech compressed:  torch.Size([1, 32, 1024])  ← 32 fixed tokens!
Speaker embedding:  torch.Size([1, 256])
Speaker projected:  torch.Size([1, 1024])
```

### Without Perceiver Resampler

```
Text tokens:        torch.Size([1, 31])
Speech tokens:      torch.Size([1, 371])
Speech embeddings:  torch.Size([1, 371, 1024])
Speech compressed:  None  ← Variable length!
Speaker embedding:  torch.Size([1, 256])
Speaker projected:  torch.Size([1, 1024])
```

## File Formats

### PyTorch Format (.pt) - For Loading

```python
import torch

# Load
data = torch.load("llm_inputs_complete.pt")

# Access
text_tokens = data["text_tokens"]
speech_compressed = data["speech_compressed"]
speaker_projected = data["speaker_projected"]
```

### JSON Format (.json) - For Inspection

```python
import json

# Load
with open("llm_inputs_complete.json") as f:
    data = json.load(f)

# Access
text_normalized = data["text_normalized"]
speech_compressed = data["speech_compressed"]
speaker_projected = data["speaker_projected"]
```

## Key Differences: Original vs Complete

| Feature | `prepare_llm_inputs.py` | `prepare_llm_inputs_with_perceiver.py` |
|---------|------------------------|--------------------------------------|
| Speech Token Embeddings | ❌ No | ✅ Yes |
| Perceiver Resampler | ❌ No | ✅ Yes |
| Compressed 32 Tokens | ❌ No | ✅ Yes |
| Complete LLM Ready | ⚠️ Partial | ✅ Full |

## Why Perceiver Matters

### Without Perceiver
- LLM receives variable-length speech context
- Different audio lengths → different input shapes
- Harder to batch and process

### With Perceiver
- **Fixed 32 tokens** regardless of audio length
- Consistent LLM input shape
- Easier batching and processing
- Better memory efficiency
- Preserves important information via cross-attention

## Memory Comparison

For 10 seconds of audio:

| Component | Tokens | Dimensions | Memory (approx) |
|-----------|--------|------------|-----------------|
| Raw Speech Tokens | 250 | 250 × 1024 × 4 bytes | ~1 MB |
| Compressed (32) | 32 | 32 × 1024 × 4 bytes | ~128 KB |
| **Savings** | **87%** | - | **~875 KB** |

## Next Steps

Once you have the prepared inputs, you can:

1. **Load into LLM** for text-to-speech generation
2. **Batch process** multiple audio files
3. **Cache compressed tokens** for faster inference
4. **Experiment** with different compression levels (modify `num_queries`)

## Troubleshooting

### Issue: Perceiver is slow
**Solution:** Perceiver runs once per audio. Cache the compressed tokens for reuse.

### Issue: Not enough compression
**Solution:** Reduce `num_queries` from 32 to 16 or 8 (may affect quality).

### Issue: Quality degradation
**Solution:** Increase `num_queries` from 32 to 64 or 128 (slower but better).

## Files

- `prepare_llm_inputs_with_perceiver.py` - Complete script with Perceiver
- `prepare_llm_inputs.py` - Basic script without Perceiver
- `chatterbox_encoders/audio/perceiver.py` - Perceiver implementation
- `LLM_INPUT_GUIDE.md` - This guide

## Summary

✅ **Use the complete script** (`prepare_llm_inputs_with_perceiver.py`) for full LLM input preparation

✅ **Perceiver Resampler is essential** for fixed-size, efficient LLM inputs

✅ **Compression from 371 → 32 tokens** saves memory and enables batching

✅ **All components ready** for manual Chatterbox LLM generation! 🎉
