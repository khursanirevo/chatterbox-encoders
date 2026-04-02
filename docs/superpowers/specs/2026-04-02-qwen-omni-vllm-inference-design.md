# Qwen3-Omni vLLM Inference Design

**Date:** 2026-04-02
**Status:** Approved
**Author:** Claude (with user approval)

## Overview

Set up vLLM offline inference for Qwen3-Omni-30B-A3B-Captioner to generate structured speech analysis captions at maximum throughput. The system processes audio files and outputs 9-field structured analysis in JSONL format.

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
   - Manages JSONL output and error logging

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
    batch_size=500,  # Optional
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
    └── qwen_omni_encoder.py  # Reuse existing
```

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
    stop=["}"],
    best_of=1,
)
```

### Batching Strategy

- **Smart batching**: Group prompts by token length
- **Max batch size**: ~500 files initially (with 432GB GPU RAM)
- **Chunking**: Process >1000 files in chunks
- **Tensor parallel**: Split each batch across 3 GPUs

### Audio Preprocessing

- Parallel audio loading using `ThreadPoolExecutor`
- Use all CPU cores for I/O-bound operations
- Cache audio encoder on GPU between batches

### Estimated Capacity

With 3x H200 @ 144GB (432GB total):
- 30B model @ 4-bit quantization: ~15GB per GPU
- Remaining ~420GB for KV cache and batch processing
- Estimated: 50-100 audio files per batch (depends on duration)

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
]
```

## Implementation Checklist

- [ ] Install vLLM dependency
- [ ] Create `vllm/` package structure
- [ ] Implement `QwenOmniAudioPreprocessor`
- [ ] Implement `VLLMInferenceEngine`
- [ ] Implement `StructuredCaptionGenerator`
- [ ] Implement main `QwenOmniCaptioner` class
- [ ] Add error handling and logging
- [ ] Test with small batch
- [ ] Performance test with large batch
- [ ] Update documentation
