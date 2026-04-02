# Qwen3-Omni vLLM Inference Design

**Date:** 2026-04-02
**Status:** Needs Revision
**Author:** Claude (with user approval)

## Overview

Set up vLLM offline inference for Qwen3-Omni-30B-A3B-Captioner to generate structured speech analysis captions at maximum throughput. The system processes audio files and outputs 9-field structured analysis in JSONL format.

## Prerequisites

1. **QwenOmniAudioEncoder Implementation** (BLOCKING)
   - The encoder must be implemented before this work
   - Currently only test file exists: `tests/test_qwen_omni_encoder.py`
   - Implementation file needed: `chatterbox_encoders/audio/qwen_omni_encoder.py`
   - **Action**: Implement encoder as part of this work or as separate prerequisite

2. **Model Availability Verification** (BLOCKING)
   - Verify `Qwen/Qwen3-Omni-30B-A3B-Captioner` exists on HuggingFace
   - If not available, identify alternative model or fine-tuning strategy
   - **Fallback**: Use base Qwen3-Omni model with custom prompt engineering

## Target Output Format

Each audio file generates a JSON object with these fields:
```json
{
  "emotion": "Emotion of the speech",
  "profile": "Speaker profile",
  "mood": "Mood of the speech",
  "speed": "Speaking speed",
  "prosody": "Prosody, rhythm",
  "pitch_timbre": "Pitch, voice quality",
  "style": "Style of utterance",
  "notes": "Other relevant notes",
  "caption": "A comprehensive caption integrating all elements"
}
```

## Architecture

### Components

1. **Audio Preprocessor** (`audio/qwen_omni_preprocessor.py`)
   - Loads and preprocesses audio files (wav, mp3, flac)
   - Resamples to 16kHz (Qwen3-Omni requirement)
   - Normalizes audio amplitude
   - Extracts audio embeddings using existing `QwenOmniAudioEncoder`

2. **vLLM Engine** (`vllm/engine.py`)
   - Manages vLLM `LLM` class with Qwen3-Omni-30B-A3B-Captioner
   - Handles 3-way tensor parallelism across 3x H200 NVL GPUs
   - Configures sampling parameters
   - Executes batched inference

3. **Caption Generator** (`vllm/caption_generator.py`)
   - Formats prompts with audio embeddings and instructions
   - Parses vLLM outputs into 9-field JSON structure
   - Validates JSON output and retries on parse failures
   - Manages JSONL output and error logging

**vLLM Integration Strategy:**

Since vLLM is designed for text-only inference, audio embeddings will be converted to a text representation:
- **Option A (Preferred)**: If Qwen3-Omni supports audio tokens natively, use special audio tokens
- **Option B (Fallback)**: Convert audio embeddings to base64-encoded text in prompt
- **Option C (Alternative)**: Use a hybrid approach where audio encoder outputs are used as context

**Prompt Template:**
```python
prompt = f"""You are a speech analysis expert. Analyze the audio features below and output ONLY valid JSON with these fields:
emotion, profile, mood, speed, prosody, pitch_timbre, style, notes, caption

Audio features (base64-encoded embeddings): {audio_embeddings_b64}

Output only valid JSON, nothing else."""
```

### Data Flow

```
Audio Files → Audio Preprocessor → vLLM Engine → Caption Generator → JSONL Output
                                                    ↓
                                               Error Log
```

1. User calls `captioner.generate_batch(audio_paths, output_path)`
2. Preprocessor loads & preprocesses all audio files
3. Audio encoder extracts embeddings for all files
4. Caption generator formats prompts (audio embeddings + instruction)
5. vLLM engine generates structured text outputs (max batch size)
6. Parser extracts JSON from each output
7. Valid results → written to `output.jsonl`
8. Failed items → logged to `errors.jsonl`

## API Design

### High-Level API

```python
from chatterbox_encoders.vllm import QwenOmniCaptioner

# Initialize
captioner = QwenOmniCaptioner(
    model_name="Qwen/Qwen3-Omni-30B-A3B-Captioner",
    tensor_parallel_size=3,
)

# Generate captions
results = captioner.generate_batch(
    audio_paths=["file1.wav", "file2.wav", ...],
    output_path="captions.jsonl",
    batch_size=100,  # Optional, auto-detected if not specified
)

# Returns statistics
# {
#     "successful": 95,
#     "failed": 5,
#     "files_per_second": 12.5,
#     "total_time_seconds": 8.0,
# }
```

### File Structure

```
chatterbox_encoders/
├── vllm/
│   ├── __init__.py          # Exports QwenOmniCaptioner
│   ├── captioner.py          # Main QwenOmniCaptioner class
│   ├── engine.py             # VLLMInferenceEngine
│   ├── caption_generator.py  # StructuredCaptionGenerator
│   └── preprocessor.py       # QwenOmniAudioPreprocessor
└── audio/
    └── qwen_omni_encoder.py  # TO BE IMPLEMENTED (prerequisite)
```

**Note**: `qwen_omni_encoder.py` must be implemented before vLLM inference can work.

## Performance Optimization

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Captioner",
    tensor_parallel_size=3,              # Use all 3 H200s
    gpu_memory_utilization=0.90,         # Aggressive memory usage
    max_model_len=8192,
    trust_remote_code=True,
    dtype="bfloat16",                    # H200 native support
    enforce_eager=False,                 # CUDA graph optimization
)

sampling_params = SamplingParams(
    temperature=0.0,                     # Deterministic (fastest)
    max_tokens=512,
    best_of=1,
)
```

### Batching Strategy

- **Smart batching**: Group prompts by token length
- **Max batch size**: Start with 50-100 files, scale up based on memory usage
- **Chunking**: Process >1000 files in chunks to avoid OOM
- **Tensor parallel**: Split each batch across 3 GPUs
- **Auto-tuning**: Monitor GPU memory and adjust batch size dynamically

### Audio Preprocessing

- Parallel audio loading using `ThreadPoolExecutor`
- Use all CPU cores for I/O-bound operations
- Cache audio encoder on GPU between batches

### Estimated Capacity

With 3x H200 @ 144GB (432GB total):
- 30B model @ 4-bit quantization: ~15GB per GPU
- Remaining ~420GB for KV cache and batch processing
- Estimated: 50-100 audio files per batch (depends on duration)

## JSON Parsing Robustness

The caption generator must handle malformed JSON from the LLM:

**Parsing Strategy:**
1. **Primary**: Use standard `json.loads()` with try/except
2. **Fallback 1**: Use `lm-format-enforcer` if available to constrain output
3. **Fallback 2**: Regex-based JSON extraction from text
4. **Retry**: If all fail, retry with lower temperature (0.0 → 0.1)
5. **Failure**: After 3 retries, log to error file

**Validation:**
- Check all 9 required fields exist
- Validate field types (all strings)
- Log warnings for missing/extra fields

## Error Handling

### Error Categories

1. **Audio Loading Errors** (corrupted files, invalid formats, missing files)
   - Action: Log to `errors.jsonl`, continue processing

2. **Encoding Errors** (audio too long, encoding failures)
   - Action: Log to `errors.jsonl`, continue processing

3. **Inference Errors** (OOM, invalid JSON, timeout)
   - Action: Retry with smaller batch (half size), then log if still fails

4. **System Errors** (GPU crash, vLLM init failure)
   - Action: Raise exception, halt entire batch

### Error Log Format

`errors.jsonl`:
```json
{"audio_path": "path/to/file.wav", "error": "File corrupted", "timestamp": "2026-04-02T14:30:00Z", "stage": "audio_loading"}
```

### Retry Strategy

For inference errors:
1. Full batch size (e.g., 100 files)
2. Retry with half batch (50 files)
3. Retry with quarter batch (25 files)
4. Skip individual files, log to errors.jsonl

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("caption_generation.log"),
        logging.StreamHandler()
    ]
)
```

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
vllm = [
    "vllm>=0.6.0",
    "lm-format-enforcer>=0.10.0",  # Optional, for JSON output enforcement
]
```

## Implementation Checklist

### Prerequisites
- [ ] Implement `QwenOmniAudioEncoder` in `chatterbox_encoders/audio/qwen_omni_encoder.py`
- [ ] Verify Qwen3-Omni-30B-A3B-Captioner model availability on HuggingFace
- [ ] Test Qwen3-Omni audio token support in vLLM

### vLLM Inference Implementation
- [ ] Install vLLM dependency
- [ ] Create `vllm/` package structure
- [ ] Implement `QwenOmniAudioPreprocessor`
- [ ] Implement `VLLMInferenceEngine`
- [ ] Implement `StructuredCaptionGenerator` with robust JSON parsing
- [ ] Implement main `QwenOmniCaptioner` class
- [ ] Add error handling and logging
- [ ] Test with small batch (10 files)
- [ ] Performance test with medium batch (100 files)
- [ ] Scale test with large batch (500+ files)
- [ ] Update documentation
