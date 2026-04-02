# Qwen3-Omni Audio Encoder Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Qwen3-Omni's pre-trained audio encoder to generate rich audio understanding embeddings, then project them to work with the existing Perceiver Resampler for token compression.

**Architecture:**
```
Audio → Qwen3-Omni Encoder (frozen) → 1280-dim → Linear Projection → 1024-dim → Perceiver → 32×1024 tokens
```

**Tech Stack:**
- Qwen3-Omni-30B-A3B-Captioner (HuggingFace transformers)
- PyTorch for projection layer
- Existing Perceiver Resampler (no changes)
- huggingface_hub for model downloading

---

## File Structure

**New Files:**
- `chatterbox_encoders/audio/qwen_omni_encoder.py` - Qwen3-Omni audio encoder wrapper class
- `chatterbox_encoders/audio/qwen_projector.py` - 1280→1024 projection layer
- `tests/test_qwen_omni_encoder.py` - Unit tests for Qwen3-Omni encoder
- `scripts/test_qwen_omni_integration.py` - End-to-end integration test script

**Modified Files:**
- `prepare_llm_inputs_with_perceiver.py` - Add `--use-qwen` flag to use Qwen3-Omni encoder
- `chatterbox_encoders/audio/__init__.py` - Export new Qwen3-Omni classes
- `pyproject.toml` - Add Qwen3-Omni dependencies (transformers, huggingface_hub)

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Qwen3-Omni dependencies to pyproject.toml**

Add to dependencies section:
```toml
dependencies = [
    # ... existing dependencies ...
    "transformers>=4.40.0",
    "huggingface-hub>=0.23.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: Packages installed successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add transformers and huggingface-hub for Qwen3-Omni"
```

---

## Task 2: Create Qwen3-Omni Audio Encoder Wrapper

**Files:**
- Create: `chatterbox_encoders/audio/qwen_omni_encoder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen_omni_encoder.py`:
```python
import pytest
import torch
import numpy as np
from pathlib import Path

def test_qwen_omni_encoder_initialization():
    """Test Qwen3-Omni encoder can be initialized."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder
    
    encoder = QwenOmniAudioEncoder(device="cpu")
    
    assert encoder is not None
    assert encoder.output_dim == 1280
    assert encoder.device == "cpu"

def test_qwen_omni_encoder_forward():
    """Test Qwen3-Omni encoder processes audio correctly."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder
    import librosa
    
    encoder = QwenOmniAudioEncoder(device="cpu")
    
    # Create dummy audio (1 second at 16kHz)
    audio = np.random.randn(16000).astype(np.float32)
    
    # Encode
    with torch.no_grad():
        embeddings = encoder.encode_audio(audio)
    
    # Check output shape: (seq_len, 1280)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 1280
    assert embeddings.shape[0] > 0  # Should have some sequence length

def test_qwen_omni_encoder_batch():
    """Test Qwen3-Omni encoder handles batch audio."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder
    
    encoder = QwenOmniAudioEncoder(device="cpu")
    
    # Create dummy audio batch (2 samples, 1 second each)
    audio_batch = [np.random.randn(16000).astype(np.float32) for _ in range(2)]
    
    # Encode batch
    with torch.no_grad():
        embeddings = encoder.encode_audio_batch(audio_batch)
    
    # Check output shape: (batch, seq_len, 1280)
    assert embeddings.ndim == 3
    assert embeddings.shape[0] == 2
    assert embeddings.shape[2] == 1280
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qwen_omni_encoder.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chatterbox_encoders.audio.qwen_omni_encoder'"

- [ ] **Step 3: Write minimal implementation**

Create `chatterbox_encoders/audio/qwen_omni_encoder.py`:
```python
"""
Qwen3-Omni Audio Encoder wrapper.

Extracts rich audio understanding embeddings from the pre-trained
Qwen3-Omni-30B-A3B-Captioner model.

Model: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner
Output: 1280-dimensional audio embeddings
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class QwenOmniAudioEncoder(nn.Module):
    """
    Wrapper for Qwen3-Omni audio encoder.
    
    Extracts 1280-dimensional audio embeddings from pre-trained Qwen3-Omni model.
    The audio encoder is frozen by default (no gradient computation).
    
    Args:
        model_name: HuggingFace model name (default: Qwen/Qwen3-Omni-30B-A3B-Captioner)
        device: Device to load model on (auto/cuda/mps/cpu)
        freeze_encoder: Whether to freeze encoder weights (default: True)
        
    Examples:
        >>> encoder = QwenOmniAudioEncoder(device="cpu")
        >>> audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        >>> embeddings = encoder.encode_audio(audio)
        >>> embeddings.shape
        torch.Size([seq_len, 1280])
    """
    
    MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
    OUTPUT_DIM = 1280
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "auto",
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = self.OUTPUT_DIM
        self.freeze_encoder = freeze_encoder
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🎧 Loading Qwen3-Omni audio encoder: {model_name}")
        logger.info(f"   Device: {self.device}")
        
        # Load model (only audio encoder part)
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            
            # Extract audio encoder from the model
            # Qwen3-Omni stores audio encoder in thinker_config.audio_config
            if hasattr(self.model, 'thinker_config'):
                audio_config = self.model.thinker_config.get('audio_config', {})
                logger.info(f"   Audio encoder d_model: {audio_config.get('d_model', 'unknown')}")
                logger.info(f"   Audio encoder layers: {audio_config.get('encoder_layers', 'unknown')}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to load Qwen3-Omni: {e}")
            raise
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            logger.info(f"   ✓ Processor loaded")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load processor: {e}")
            self.processor = None
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info(f"   ✓ Encoder frozen")
        else:
            logger.info(f"   ⚠️ Encoder trainable (fine-tuning mode)")
        
        logger.info(f"   ✓ Qwen3-Omni audio encoder ready")
        logger.info(f"   Output dimension: {self.output_dim}")
    
    def encode_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode audio to embeddings.
        
        Args:
            audio: Audio waveform (samples,) - float32, 16kHz
            sample_rate: Sample rate (default: 16000)
            return_tensor: Whether to return tensor (True) or numpy array (False)
        
        Returns:
            Audio embeddings: (seq_len, 1280)
        
        Examples:
            >>> encoder = QwenOmniAudioEncoder(device="cpu")
            >>> audio = np.random.randn(16000).astype(np.float32)
            >>> emb = encoder.encode_audio(audio)
            >>> emb.shape
            torch.Size([seq_len, 1280])
        """
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, samples)
        
        # Extract audio features using the model's audio encoder
        with torch.no_grad():
            # The Qwen3-Omni model processes audio through its audio encoder
            # We need to call it with the appropriate inputs
            if self.processor is not None:
                # Use processor if available
                inputs = self.processor(
                    audio=audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract audio embeddings
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get audio encoder hidden states
                # This may vary depending on the model structure
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Use the last hidden state from audio encoder
                    audio_hidden = outputs.hidden_states[-1]  # (batch, seq, dim)
                else:
                    # Fallback to last_hidden_state
                    audio_hidden = outputs.last_hidden_state
            else:
                # Direct processing without processor
                # This is a simplified approach - may need adjustment
                audio_hidden = self.model.audio_forward(
                    audio_tensor,
                    sampling_rate=sample_rate,
                )
        
        # Remove batch dimension if single input
        embeddings = audio_hidden.squeeze(0)  # (seq_len, 1280)
        
        if not return_tensor:
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def encode_audio_batch(
        self,
        audio_batch: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Encode batch of audio to embeddings.
        
        Args:
            audio_batch: List of audio waveforms, each (samples,)
            sample_rate: Sample rate (default: 16000)
        
        Returns:
            Audio embeddings: (batch, seq_len, 1280)
        
        Examples:
            >>> encoder = QwenOmniAudioEncoder(device="cpu")
            >>> audio_batch = [np.random.randn(16000).astype(np.float32) for _ in range(2)]
            >>> emb = encoder.encode_audio_batch(audio_batch)
            >>> emb.shape
            torch.Size([2, seq_len, 1280])
        """
        # Encode each audio individually
        embeddings_list = []
        for audio in audio_batch:
            emb = self.encode_audio(audio, sample_rate, return_tensor=True)
            embeddings_list.append(emb)
        
        # Stack with padding (if different lengths)
        from torch.nn.utils.rnn import pad_sequence
        embeddings_padded = pad_sequence(
            embeddings_list,
            batch_first=True,
            padding_value=0.0,
        )
        
        return embeddings_padded
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch Module interface.
        
        Args:
            audio: Audio tensor (batch, samples)
        
        Returns:
            Embeddings: (batch, seq_len, 1280)
        """
        # Convert tensor to numpy for encode_audio
        audio_np = audio.squeeze(0).cpu().numpy()
        
        # Encode
        embeddings = self.encode_audio(audio_np, return_tensor=True)
        
        # Add batch dimension back
        return embeddings.unsqueeze(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qwen_omni_encoder.py -v`
Expected: Tests may fail due to model structure specifics - iterate on implementation

- [ ] **Step 5: Commit**

```bash
git add chatterbox_encoders/audio/qwen_omni_encoder.py tests/test_qwen_omni_encoder.py
git commit -m "feat: add Qwen3-Omni audio encoder wrapper"
```

---

## Task 3: Create Projection Layer (1280 → 1024)

**Files:**
- Create: `chatterbox_encoders/audio/qwen_projector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_qwen_omni_encoder.py`:
```python
def test_qwen_projector():
    """Test Qwen3-Omni projection layer."""
    from chatterbox_encoders.audio.qwen_projector import QwenProjector
    import torch
    
    projector = QwenProjector(device="cpu")
    
    # Create dummy Qwen embeddings (batch, seq_len, 1280)
    qwen_emb = torch.randn(1, 100, 1280)
    
    # Project
    projected = projector(qwen_emb)
    
    # Check output shape: (batch, seq_len, 1024)
    assert projected.shape == (1, 100, 1024)

def test_qwen_projector_with_perceiver():
    """Test full pipeline: Qwen → Projector → Perceiver."""
    from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder
    from chatterbox_encoders.audio.qwen_projector import QwenProjector
    from chatterbox_encoders.audio.perceiver import PerceiverResampler
    import numpy as np
    
    # Initialize components
    encoder = QwenOmniAudioEncoder(device="cpu")
    projector = QwenProjector(device="cpu")
    perceiver = PerceiverResampler(num_queries=32, query_dim=1024, embedding_dim=1024)
    perceiver = perceiver.to("cpu")
    
    # Create dummy audio
    audio = np.random.randn(16000).astype(np.float32)
    
    # Full pipeline
    with torch.no_grad():
        # Encode with Qwen
        qwen_emb = encoder.encode_audio(audio)  # (seq_len, 1280)
        qwen_emb = qwen_emb.unsqueeze(0)  # (1, seq_len, 1280)
        
        # Project to 1024
        projected = projector(qwen_emb)  # (1, seq_len, 1024)
        
        # Compress with Perceiver
        compressed = perceiver(projected)  # (1, 32, 1024)
    
    # Check final output
    assert compressed.shape == (1, 32, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qwen_omni_encoder.py::test_qwen_projector -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chatterbox_encoders.audio.qwen_projector'"

- [ ] **Step 3: Write minimal implementation**

Create `chatterbox_encoders/audio/qwen_projector.py`:
```python
"""
Projection layer for Qwen3-Omni embeddings.

Projects 1280-dimensional Qwen3-Omni audio embeddings to 1024 dimensions
for compatibility with the Perceiver Resampler.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QwenProjector(nn.Module):
    """
    Projection layer for Qwen3-Omni embeddings.
    
    Projects 1280-dimensional Qwen3-Omni audio embeddings to 1024 dimensions
    for compatibility with the Perceiver Resampler.
    
    Args:
        input_dim: Input dimension (default: 1280 for Qwen3-Omni)
        output_dim: Output dimension (default: 1024 for Perceiver)
        hidden_dim: Optional hidden dimension for 2-layer projection (default: None)
        dropout: Dropout rate (default: 0.1)
        
    Examples:
        >>> projector = QwenProjector()
        >>> qwen_emb = torch.randn(1, 100, 1280)
        >>> projected = projector(qwen_emb)
        >>> projected.shape
        torch.Size([1, 100, 1024])
    """
    
    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        if hidden_dim is None:
            # Single linear layer projection
            self.projection = nn.Linear(input_dim, output_dim)
            logger.info(f"QwenProjector: {input_dim} → {output_dim} (single layer)")
        else:
            # Two-layer projection with hidden layer
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout),
            )
            logger.info(f"QwenProjector: {input_dim} → {hidden_dim} → {output_dim} (two-layer)")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Qwen3-Omni embeddings to Perceiver dimension.
        
        Args:
            x: Qwen3-Omni embeddings (batch, seq_len, 1280)
        
        Returns:
            Projected embeddings (batch, seq_len, 1024)
        
        Examples:
            >>> projector = QwenProjector()
            >>> emb = torch.randn(1, 100, 1280)
            >>> out = projector(emb)
            >>> out.shape
            torch.Size([1, 100, 1024])
        """
        return self.projection(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qwen_omni_encoder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chatterbox_encoders/audio/qwen_projector.py tests/test_qwen_omni_encoder.py
git commit -m "feat: add Qwen3-Omni projection layer (1280 → 1024)"
```

---

## Task 4: Update Audio Module Exports

**Files:**
- Modify: `chatterbox_encoders/audio/__init__.py`

- [ ] **Step 1: Export new classes**

Add to `chatterbox_encoders/audio/__init__.py`:
```python
from chatterbox_encoders.audio.qwen_omni_encoder import QwenOmniAudioEncoder
from chatterbox_encoders.audio.qwen_projector import QwenProjector

__all__ = [
    # ... existing exports ...
    "QwenOmniAudioEncoder",
    "QwenProjector",
]
```

- [ ] **Step 2: Verify exports work**

Run: `uv run python -c "from chatterbox_encoders.audio import QwenOmniAudioEncoder, QwenProjector; print('✓ Imports work')"`
Expected: ✓ Imports work

- [ ] **Step 3: Commit**

```bash
git add chatterbox_encoders/audio/__init__.py
git commit -m "feat: export Qwen3-Omni classes from audio module"
```

---

## Task 5: Integrate with LLM Input Preparation

**Files:**
- Modify: `prepare_llm_inputs_with_perceiver.py`

- [ ] **Step 1: Add Qwen3-Omni option to CompleteLLMInputPreparer**

Modify the `__init__` method in `prepare_llm_inputs_with_perceiver.py`:
```python
def __init__(
    self,
    device: str = "auto",
    ve_checkpoint: str = None,
    tokenizer_path: str = None,
    load_perceiver: bool = True,
    use_qwen_encoder: bool = False,  # NEW PARAMETER
):
    """
    Initialize complete LLM input preparer.
    
    Args:
        device: Device to use (auto/cuda/mps/cpu)
        ve_checkpoint: Path to voice encoder checkpoint
        tokenizer_path: Path to text tokenizer
        load_perceiver: Whether to load Perceiver Resampler
        use_qwen_encoder: Whether to use Qwen3-Omni encoder (NEW)
    """
    # ... existing initialization ...
    
    # NEW: Initialize Qwen3-Omni encoder if requested
    self.use_qwen_encoder = use_qwen_encoder
    if use_qwen_encoder:
        logger.info("🎧 Loading Qwen3-Omni audio encoder...")
        from chatterbox_encoders.audio import QwenOmniAudioEncoder, QwenProjector
        
        self.qwen_encoder = QwenOmniAudioEncoder(device=self.device)
        self.qwen_projector = QwenProjector()
        self.qwen_projector = self.qwen_projector.to(self.device)
        self.qwen_projector.eval()
        logger.info(f"   ✓ Qwen3-Omni + Projector ready")
    else:
        self.qwen_encoder = None
        self.qwen_projector = None
```

- [ ] **Step 2: Update speech token preparation to use Qwen3-Omni**

Modify the `prepare_speech_tokens_with_embeddings` method:
```python
def prepare_speech_tokens_with_embeddings(
    self,
    audio_path: str,
    max_duration: float = 30.0,
) -> dict:
    """
    Prepare speech tokens AND embeddings from reference audio.
    
    Args:
        audio_path: Path to reference audio file
        max_duration: Maximum audio duration in seconds
    
    Returns:
        dict: {
            "tokens": torch.Tensor (1, num_tokens),
            "embeddings": torch.Tensor (1, num_tokens, 1024),
            "compressed": torch.Tensor (1, 32, 1024) or None,
            "audio": np.ndarray,
            "sample_rate": int,
            "duration": float,
            "qwen_embeddings": torch.Tensor or None,  # NEW
        }
    """
    logger.info(f"🎵 Preparing speech tokens from: {audio_path}")
    
    # Load audio at 16kHz
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Trim to max duration
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        logger.info(f"   Trimmed to {max_duration}s")
    
    # Ensure float32
    audio = audio.astype(np.float32)
    
    # NEW: Extract Qwen3-Omni embeddings if enabled
    qwen_embeddings = None
    if self.use_qwen_encoder:
        with torch.no_grad():
            # Encode with Qwen3-Omni
            qwen_emb = self.qwen_encoder.encode_audio(audio)  # (seq_len, 1280)
            qwen_emb = qwen_emb.unsqueeze(0)  # (1, seq_len, 1280)
            
            # Project to 1024
            qwen_embeddings = self.qwen_projector(qwen_emb)  # (1, seq_len, 1024)
            qwen_embeddings = qwen_embeddings.to(self.device)
        
        logger.info(f"   ✓ Qwen embeddings: {qwen_embeddings.shape}")
    
    # Tokenize with S3Tokenizer
    with torch.no_grad():
        tokens, lengths = self.s3_tokenizer.forward([audio])
    
    # Use Qwen embeddings if available, otherwise use learned embeddings
    if qwen_embeddings is not None:
        # Truncate or pad Qwen embeddings to match token sequence length
        # This is a simplified approach - may need refinement
        embeddings = qwen_embeddings
    else:
        # Convert to embeddings using learned embedding layer
        with torch.no_grad():
            embeddings = self.speech_embedding(tokens)  # (1, T, 1024)
    
    # Compress with Perceiver Resampler (if loaded)
    compressed = None
    if self.perceiver is not None:
        with torch.no_grad():
            compressed = self.perceiver(embeddings)  # (1, 32, 1024)
        logger.info(f"   ✓ Compressed: {embeddings.shape} → {compressed.shape}")
    
    result = {
        "tokens": tokens.to(self.device),
        "embeddings": embeddings.to(self.device),
        "compressed": compressed.to(self.device) if compressed is not None else None,
        "audio": audio,
        "sample_rate": sr,
        "duration": len(audio) / sr,
        "qwen_embeddings": qwen_embeddings,  # NEW
    }
    
    logger.info(f"   ✓ Tokens shape: {result['tokens'].shape}")
    logger.info(f"   ✓ Embeddings shape: {result['embeddings'].shape}")
    logger.info(f"   ✓ Duration: {result['duration']:.2f}s")
    
    return result
```

- [ ] **Step 3: Add CLI flag for Qwen3-Omni encoder**

Update the argument parser in `main()`:
```python
parser.add_argument(
    "--use-qwen",
    action="store_true",
    help="Use Qwen3-Omni audio encoder instead of learned embeddings"
)
```

And pass to CompleteLLMInputPreparer:
```python
preparer = CompleteLLMInputPreparer(
    device=args.device,
    ve_checkpoint=args.ve_checkpoint,
    load_perceiver=not args.no_perceiver,
    use_qwen_encoder=args.use_qwen,  # NEW
)
```

- [ ] **Step 4: Test the integration**

Run:
```bash
uv run python prepare_llm_inputs_with_perceiver.py \
    --text "Hello world" \
    --audio /path/to/test.wav \
    --use-qwen \
    --output test_qwen.pt
```
Expected: Script runs successfully, Qwen embeddings are extracted and used

- [ ] **Step 5: Commit**

```bash
git add prepare_llm_inputs_with_perceiver.py
git commit -m "feat: add Qwen3-Omni encoder option to LLM input preparation"
```

---

## Task 6: Create End-to-End Integration Test Script

**Files:**
- Create: `scripts/test_qwen_omni_integration.py`

- [ ] **Step 1: Create integration test script**

Create `scripts/test_qwen_omni_integration.py`:
```python
"""
End-to-end integration test for Qwen3-Omni audio encoder.

Tests the full pipeline:
Audio → Qwen3-Omni → Projector → Perceiver → 32×1024 tokens
"""

import logging
import sys
from pathlib import Path

import torch
import numpy as np

from chatterbox_encoders.audio import (
    QwenOmniAudioEncoder,
    QwenProjector,
    PerceiverResampler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_qwen_omni_integration():
    """Test full Qwen3-Omni integration pipeline."""
    
    logger.info("="*60)
    logger.info("QWEN3-OMNI INTEGRATION TEST")
    logger.info("="*60)
    
    # Initialize components
    logger.info("Step 1: Initialize components")
    encoder = QwenOmniAudioEncoder(device="cpu")
    projector = QwenProjector()
    perceiver = PerceiverResampler(
        num_queries=32,
        query_dim=1024,
        embedding_dim=1024,
    )
    
    logger.info(f"   ✓ Qwen3-Omni encoder: {encoder.output_dim} dims")
    logger.info(f"   ✓ Projector: 1280 → 1024")
    logger.info(f"   ✓ Perceiver: variable → 32 tokens")
    
    # Create dummy audio
    logger.info("\nStep 2: Create dummy audio (1 second)")
    audio = np.random.randn(16000).astype(np.float32)
    logger.info(f"   ✓ Audio shape: {audio.shape}")
    
    # Encode with Qwen3-Omni
    logger.info("\nStep 3: Encode with Qwen3-Omni")
    with torch.no_grad():
        qwen_emb = encoder.encode_audio(audio)
        qwen_emb = qwen_emb.unsqueeze(0)  # Add batch dim
    logger.info(f"   ✓ Qwen embeddings: {qwen_emb.shape}")
    
    # Project to 1024
    logger.info("\nStep 4: Project to 1024 dimensions")
    with torch.no_grad():
        projected = projector(qwen_emb)
    logger.info(f"   ✓ Projected embeddings: {projected.shape}")
    
    # Compress with Perceiver
    logger.info("\nStep 5: Compress with Perceiver Resampler")
    with torch.no_grad():
        compressed = perceiver(projected)
    logger.info(f"   ✓ Compressed tokens: {compressed.shape}")
    
    # Verify output
    logger.info("\nStep 6: Verify output")
    assert compressed.shape == (1, 32, 1024), f"Unexpected shape: {compressed.shape}"
    assert not torch.isnan(compressed).any(), "Output contains NaN"
    assert not torch.isinf(compressed).any(), "Output contains Inf"
    logger.info(f"   ✓ Output shape correct: {compressed.shape}")
    logger.info(f"   ✓ No NaN or Inf values")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*60)
    
    return True


def test_comparison_with_learned_embeddings():
    """Compare Qwen3-Omni embeddings vs learned embeddings."""
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON TEST: Qwen vs Learned Embeddings")
    logger.info("="*60)
    
    # This test would compare the quality of embeddings
    # For now, just verify both pipelines work
    logger.info("   (Placeholder for future comparison tests)")
    
    return True


if __name__ == "__main__":
    try:
        # Run tests
        test_qwen_omni_integration()
        test_comparison_with_learned_embeddings()
        
        logger.info("\n✅ Integration test complete!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
```

- [ ] **Step 2: Run integration test**

Run: `uv run python scripts/test_qwen_omni_integration.py`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add scripts/test_qwen_omni_integration.py
git commit -m "test: add Qwen3-Omni integration test script"
```

---

## Task 7: Documentation

**Files:**
- Create: `docs/qwen_omni_integration.md`

- [ ] **Step 1: Write documentation**

Create `docs/qwen_omni_integration.md`:
```markdown
# Qwen3-Omni Audio Encoder Integration

## Overview

This document describes the integration of Qwen3-Omni's pre-trained audio encoder for generating rich audio understanding embeddings.

## Architecture

```
Audio → Qwen3-Omni Encoder (frozen) → 1280-dim → Linear Projection → 1024-dim → Perceiver → 32×1024 tokens
```

## Usage

### Basic Usage

```python
from chatterbox_encoders.audio import QwenOmniAudioEncoder, QwenProjector, PerceiverResampler

# Initialize components
encoder = QwenOmniAudioEncoder(device="cuda")
projector = QwenProjector()
perceiver = PerceiverResampler(num_queries=32)

# Process audio
qwen_emb = encoder.encode_audio(audio)  # (seq_len, 1280)
projected = projector(qwen_emb.unsqueeze(0))  # (1, seq_len, 1024)
compressed = perceiver(projected)  # (1, 32, 1024)
```

### CLI Usage

```bash
python prepare_llm_inputs_with_perceiver.py \
    --text "Hello world" \
    --audio reference.wav \
    --use-qwen \
    --output llm_inputs_qwen.pt
```

## Model Details

- **Model:** Qwen/Qwen3-Omni-30B-A3B-Captioner
- **Audio Encoder Output:** 1280 dimensions
- **Encoder Layers:** 32
- **Attention Heads:** 20
- **Sample Rate:** 16kHz
- **Mel Bins:** 128

## Benefits

1. **Pre-trained Audio Understanding:** Leverages Qwen3-Omni's training on diverse audio tasks
2. **Rich Representations:** 1280-dim embeddings capture detailed audio features
3. **Minimal Training:** Only projection layer (1280→1024) needs to be learned
4. **Frozen Encoder:** No gradient computation through the large encoder model

## Performance Considerations

- **Model Size:** Qwen3-Omni-30B is large (~30B parameters total)
- **Memory:** Requires significant GPU memory for inference
- **Inference Speed:** Slower than learned embeddings due to model size
- **Recommendation:** Use for experimentation/benchmarking, consider smaller variants for production

## Future Work

- [ ] Fine-tuning Qwen3-Omni encoder on speech token data
- [ ] Comparing embedding quality vs learned embeddings
- [ ] Benchmarking reconstruction quality
- [ ] Exploring smaller Qwen3-Omni variants
```

- [ ] **Step 2: Commit**

```bash
git add docs/qwen_omni_integration.md
git commit -m "docs: add Qwen3-Omni integration documentation"
```

---

## Task 8: Final Verification

**Files:**
- All files

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/test_qwen_omni_encoder.py -v`
Expected: All tests pass

- [ ] **Step 2: Run integration test**

Run: `uv run python scripts/test_qwen_omni_integration.py`
Expected: Integration test passes

- [ ] **Step 3: Run end-to-end test with real audio**

Run:
```bash
uv run python prepare_llm_inputs_with_perceiver.py \
    --text "Testing Qwen3-Omni integration" \
    --audio /path/to/real_audio.wav \
    --use-qwen \
    --output test_final.pt
```
Expected: Script completes successfully, output file contains Qwen embeddings

- [ ] **Step 4: Verify linting**

Run: `uv run ruff check chatterbox_encoders/audio/qwen_*.py tests/test_qwen_omni_encoder.py`
Expected: No linting errors

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete Qwen3-Omni audio encoder integration"
```

---

## Summary

This plan implements the Qwen3-Omni audio encoder integration with:

1. **QwenOmniAudioEncoder:** Wrapper for pre-trained Qwen3-Omni model (1280-dim output)
2. **QwenProjector:** Linear projection layer (1280 → 1024)
3. **Integration:** Updated `prepare_llm_inputs_with_perceiver.py` with `--use-qwen` flag
4. **Tests:** Unit tests and integration tests
5. **Documentation:** Complete usage guide

The architecture leverages pre-trained audio understanding while maintaining compatibility with the existing Perceiver Resampler.
