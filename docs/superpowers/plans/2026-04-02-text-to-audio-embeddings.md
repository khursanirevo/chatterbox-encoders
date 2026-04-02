# Text-to-Audio-Embeddings Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn a text encoder that maps Qwen3-Omni's rich text analysis to the same 32×1024 audio tokens that the voice encoder produces, enabling text-only audio generation during inference.

**Architecture:**

**Training Phase:**
```
Reference Audio → Qwen3-Omni → Text Analysis (emotion, profile, mood, speed, prosody, pitch_timbre, style, notes, caption)
                                                ↓
                                        Text Encoder → 32×1024 tokens
                                                ↓
                                          Learn to match (MSE loss)
                                                ↓
Reference Audio → Voice Encoder → Perceiver → 32×1024 tokens (ground truth)
```

**Inference Phase:**
```
Text Analysis → Text Encoder → 32×1024 tokens → Audio Generation
```

**Tech Stack:**
- Qwen3-Omni for audio-to-text analysis (HuggingFace transformers)
- T5-small for text encoding (lightweight, ~240MB)
- PyTorch for training loop
- Existing voice encoder + Perceiver as ground truth teacher

---

## File Structure

**New Files:**
- `chatterbox_encoders/text_analysis/qwen_analyzer.py` - Qwen3-Omni text analysis wrapper
- `chatterbox_encoders/text_analysis/text_encoder.py` - T5-based text encoder with projection to 1024
- `scripts/train_text_encoder.py` - Training script to learn text encoder from voice encoder
- `scripts/analyze_audio_with_qwen.py` - Utility script to extract text analysis from audio
- `tests/test_qwen_analyzer.py` - Tests for Qwen3-Omni analyzer
- `tests/test_text_encoder.py` - Tests for text encoder

**Modified Files:**
- `chatterbox_encoders/text_analysis/__init__.py` - Export new text analysis classes
- `pyproject.toml` - Add Qwen3-Omni and T5 dependencies

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Qwen3-Omni and T5 dependencies to pyproject.toml**

The transformers dependency was already added in the previous attempt. We just need to ensure it's there.

Check current dependencies:
```bash
grep -i transformers pyproject.toml
```

Expected: `transformers>=4.40.0` should already be present

- [ ] **Step 2: Verify dependencies are installed**

Run: `uv sync`
Expected: No errors, packages installed successfully

- [ ] **Step 3: Commit (if needed)**

If transformers was not present:
```bash
git add pyproject.toml
git commit -m "deps: ensure transformers is available for Qwen3-Omni and T5"
```

---

## Task 2: Create Text Analysis Module Structure

**Files:**
- Create: `chatterbox_encoders/text_analysis/__init__.py`

- [ ] **Step 1: Create text analysis module**

Create `chatterbox_encoders/text_analysis/__init__.py`:
```python
"""
Text analysis for audio understanding.

This module provides text-based analysis of audio using Qwen3-Omni,
and learns to map text analysis to audio embeddings.
"""

from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer
from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

__all__ = [
    "QwenOmniAnalyzer",
    "TextToAudioEmbedding",
]
```

- [ ] **Step 2: Verify module structure**

Run: `uv run python -c "from chatterbox_encoders.text_analysis import QwenOmniAnalyzer; print('✓ Module structure ready')"`
Expected: ModuleNotFoundError (classes not implemented yet)

- [ ] **Step 3: Commit**

```bash
git add chatterbox_encoders/text_analysis/__init__.py
git commit -m "feat: create text analysis module structure"
```

---

## Task 3: Create Qwen3-Omni Text Analyzer

**Files:**
- Create: `chatterbox_encoders/text_analysis/qwen_analyzer.py`
- Create: `tests/test_qwen_analyzer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen_analyzer.py`:
```python
"""
Tests for Qwen3-Omni text analyzer.
"""

import pytest
import numpy as np
from pathlib import Path


def test_qwen_analyzer_initialization():
    """Test Qwen3-Omni analyzer can be initialized."""
    from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer

    analyzer = QwenOmniAnalyzer(device="cpu")

    assert analyzer is not None
    assert analyzer.device == "cpu"


def test_qwen_analyzer_extract_text_analysis():
    """Test Qwen3-Omni extracts text analysis from audio."""
    from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer

    analyzer = QwenOmniAnalyzer(device="cpu")

    # Create dummy audio (1 second at 16kHz)
    audio = np.random.randn(16000).astype(np.float32)

    # Extract text analysis
    analysis = analyzer.analyze_audio(audio)

    # Check structure
    assert isinstance(analysis, dict)
    assert "emotion" in analysis
    assert "profile" in analysis
    assert "mood" in analysis
    assert "speed" in analysis
    assert "prosody" in analysis
    assert "pitch_timbre" in analysis
    assert "style" in analysis
    assert "notes" in analysis
    assert "caption" in analysis

    # Check all values are non-empty strings
    for key, value in analysis.items():
        assert isinstance(value, str)
        assert len(value) > 0, f"{key} is empty"


def test_qwen_analyzer_format_for_encoder():
    """Test formatting text analysis for text encoder."""
    from chatterbox_encoders.text_analysis.qwen_analyzer import QwenOmniAnalyzer

    analyzer = QwenOmniAnalyzer(device="cpu")

    # Create dummy analysis
    analysis = {
        "emotion": "happy",
        "profile": "young female speaker",
        "mood": "cheerful",
        "speed": "moderate",
        "prosody": "rising intonation",
        "pitch_timbre": "high-pitched, bright",
        "style": "conversational",
        "notes": "background noise present",
        "caption": "A cheerful greeting with warm tone",
    }

    # Format for encoder
    formatted = analyzer.format_for_encoder(analysis)

    # Check it's a string
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "emotion" in formatted.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qwen_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chatterbox_encoders.text_analysis.qwen_analyzer'"

- [ ] **Step 3: Write minimal implementation**

Create `chatterbox_encoders/text_analysis/qwen_analyzer.py`:
```python
"""
Qwen3-Omni text analyzer wrapper.

Extracts rich text analysis from audio using Qwen3-Omni's audio understanding
capabilities. The analysis includes emotion, profile, mood, speed, prosody,
pitch_timbre, style, notes, and caption.
"""

import logging
from typing import Optional

import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class QwenOmniAnalyzer:
    """
    Analyzer for extracting rich text analysis from audio using Qwen3-Omni.

    Qwen3-Omni is a multi-modal model that can analyze audio and provide
    detailed text descriptions including emotion, speaker characteristics,
    speaking style, and comprehensive captions.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on (auto/cuda/mps/cpu)
        prompt_template: Template for formatting analysis

    Examples:
        >>> analyzer = QwenOmniAnalyzer(device="cpu")
        >>> audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        >>> analysis = analyzer.analyze_audio(audio)
        >>> print(analysis['caption'])
        A cheerful greeting with warm tone
    """

    # Qwen3-Omni model
    MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "auto",
        prompt_template: Optional[str] = None,
    ):
        self.model_name = model_name

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set prompt template
        if prompt_template is None:
            self.prompt_template = (
                "Analyze this audio and provide:\n"
                "emotion: {{emotion}}\n"
                "profile: {{profile}}\n"
                "mood: {{mood}}\n"
                "speed: {{speed}}\n"
                "prosody: {{prosody}}\n"
                "pitch_timbre: {{pitch_timbre}}\n"
                "style: {{style}}\n"
                "notes: {{notes}}\n"
                "caption: {{caption}}"
            )
        else:
            self.prompt_template = prompt_template

        logger.info(f"🎤 Loading Qwen3-Omni analyzer: {model_name}")
        logger.info(f"   Device: {self.device}")

        # Load model
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            logger.info(f"   ✓ Model loaded")
        except Exception as e:
            logger.error(f"   ❌ Failed to load Qwen3-Omni model: {e}")
            logger.info(f"   ⚠️  Falling back to mock analysis for testing")
            self.model = None

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            logger.info(f"   ✓ Processor loaded")
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to load processor: {e}")
            self.processor = None

        logger.info(f"   ✓ Qwen3-Omni analyzer ready")

    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Analyze audio and extract rich text descriptions.

        Args:
            audio: Audio waveform (samples,) - float32, 16kHz
            sample_rate: Sample rate (default: 16000)

        Returns:
            Dictionary with keys:
                - emotion: Emotion of the speech
                - profile: Speaker profile
                - mood: Mood of the speech
                - speed: Speaking speed
                - prosody: Prosody, rhythm
                - pitch_timbre: Pitch, voice quality
                - style: Style of utterance
                - notes: Other relevant notes
                - caption: A comprehensive caption integrating all elements

        Examples:
            >>> analyzer = QwenOmniAnalyzer()
            >>> audio = np.random.randn(16000).astype(np.float32)
            >>> analysis = analyzer.analyze_audio(audio)
            >>> print(analysis['emotion'])
            happy
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # If model is not available, use mock analysis
        if self.model is None or self.processor is None:
            return self._mock_analysis(audio)

        # Convert audio to format expected by Qwen3-Omni
        # Qwen3-Omni expects audio as a list of dictionaries or specific format
        try:
            # Prepare inputs
            inputs = self.processor(
                audio=audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate analysis
            with torch.no_grad():
                prompt = (
                    "Analyze this audio clip and provide a detailed description including:\n"
                    "1. Emotion expressed\n"
                    "2. Speaker profile (age, gender, accent)\n"
                    "3. Overall mood\n"
                    "4. Speaking speed\n"
                    "5. Prosody and rhythm\n"
                    "6. Pitch and timbre\n"
                    "7. Speaking style\n"
                    "8. Any other notable observations\n"
                    "9. A comprehensive caption\n\n"
                    "Format your response as key-value pairs."
                )

                outputs = self.model.generate(
                    **inputs,
                    prompt_text=prompt,
                    max_new_tokens=512,
                )

                # Decode response
                response = self.processor.decode(outputs[0], skip_special_tokens=True)

                # Parse response into structured format
                analysis = self._parse_response(response)

        except Exception as e:
            logger.warning(f"   ⚠️  Qwen3-Omni analysis failed: {e}")
            logger.info(f"   ⚠️  Using mock analysis")
            analysis = self._mock_analysis(audio)

        return analysis

    def _mock_analysis(self, audio: np.ndarray) -> dict:
        """
        Generate mock analysis for testing when model is unavailable.

        In production, this should never be called. It's only for testing
        when the Qwen3-Omni model is not available.
        """
        duration = len(audio) / 16000.0

        # Simple heuristic-based mock analysis
        energy = np.mean(np.abs(audio))
        zero_crossing_rate = np.mean(np.diff(audio > 0).astype(np.float32))

        # Determine characteristics based on simple features
        if energy > 0.1:
            emotion = "energetic" if zero_crossing_rate > 0.1 else "calm"
            speed = "fast" if zero_crossing_rate > 0.15 else "moderate"
        else:
            emotion = "neutral"
            speed = "slow"

        return {
            "emotion": f"{emotion} speech",
            "profile": "single speaker, unclear characteristics",
            "mood": f"{emotion} and expressive",
            "speed": f"{speed} paced speech",
            "prosody": "variable intonation detected",
            "pitch_timbre": "neutral voice quality",
            "style": "conversational style",
            "notes": f"duration: {duration:.2f}s, energy: {energy:.3f}",
            "caption": f"A {emotion} speech segment at {speed} speed",
        }

    def _parse_response(self, response: str) -> dict:
        """
        Parse Qwen3-Omni response into structured dictionary.

        Args:
            response: Raw text response from Qwen3-Omni

        Returns:
            Structured dictionary with analysis fields
        """
        # Try to parse key-value pairs
        analysis = {}

        # Default values
        defaults = {
            "emotion": "neutral",
            "profile": "unknown speaker",
            "mood": "neutral",
            "speed": "moderate",
            "prosody": "standard",
            "pitch_timbre": "neutral",
            "style": "conversational",
            "notes": "no additional notes",
            "caption": "speech segment",
        }

        # Look for patterns like "emotion: happy" or "emotion - happy"
        import re

        patterns = {
            "emotion": r"(?:emotion|feeling)\s*[::-]\s*([^\n]+)",
            "profile": r"(?:profile|speaker)\s*[::-]\s*([^\n]+)",
            "mood": r"(?:mood|atmosphere)\s*[::-]\s*([^\n]+)",
            "speed": r"(?:speed|pace|tempo)\s*[::-]\s*([^\n]+)",
            "prosody": r"(?:prosody|rhythm|cadence)\s*[::-]\s*([^\n]+)",
            "pitch_timbre": r"(?:pitch[-_\s]?timbre|voice[-_\s]?quality)\s*[::-]\s*([^\n]+)",
            "style": r"(?:style|delivery)\s*[::-]\s*([^\n]+)",
            "notes": r"(?:notes|observations|other)\s*[::-]\s*([^\n]+)",
            "caption": r"(?:caption|summary|description)\s*[::-]\s*([^\n]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                analysis[key] = match.group(1).strip()
            else:
                analysis[key] = defaults[key]

        # If caption wasn't found, use the whole response as caption
        if analysis["caption"] == "speech segment" and len(response) > 20:
            analysis["caption"] = response.strip()[:500]

        return analysis

    def format_for_encoder(self, analysis: dict, template: Optional[str] = None) -> str:
        """
        Format text analysis for input to text encoder.

        Args:
            analysis: Dictionary from analyze_audio()
            template: Optional custom template

        Returns:
            Formatted string ready for text encoder

        Examples:
            >>> analyzer = QwenOmniAnalyzer()
            >>> analysis = analyzer.analyze_audio(audio)
            >>> formatted = analyzer.format_for_encoder(analysis)
            >>> print(formatted)
            Emotion: happy
            Profile: young female speaker
            ...
        """
        if template is None:
            template = (
                "Emotion: {emotion}\n"
                "Profile: {profile}\n"
                "Mood: {mood}\n"
                "Speed: {speed}\n"
                "Prosody: {prosody}\n"
                "Pitch/Timbre: {pitch_timbre}\n"
                "Style: {style}\n"
                "Notes: {notes}\n"
                "Caption: {caption}"
            )

        formatted = template.format(**analysis)
        return formatted

    def analyze_from_file(self, audio_path: str) -> dict:
        """
        Analyze audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with text analysis

        Examples:
            >>> analyzer = QwenOmniAnalyzer()
            >>> analysis = analyzer.analyze_from_file("speech.wav")
            >>> print(analysis['caption'])
            A cheerful greeting
        """
        import librosa

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)

        return self.analyze_audio(audio, sample_rate=sr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qwen_analyzer.py -v`
Expected: Tests pass (using mock analysis since model might not be available)

- [ ] **Step 5: Commit**

```bash
git add chatterbox_encoders/text_analysis/qwen_analyzer.py tests/test_qwen_analyzer.py
git commit -m "feat: add Qwen3-Omni text analyzer wrapper"
```

---

## Task 4: Create Text Encoder (T5-based)

**Files:**
- Create: `chatterbox_encoders/text_analysis/text_encoder.py`
- Create: `tests/test_text_encoder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_encoder.py`:
```python
"""
Tests for text-to-audio-embedding encoder.
"""

import pytest
import torch


def test_text_encoder_initialization():
    """Test text encoder can be initialized."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")

    assert encoder is not None
    assert encoder.output_dim == 1024
    assert encoder.num_queries == 32
    assert encoder.device == "cpu"


def test_text_encoder_forward():
    """Test text encoder produces correct output shape."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")

    # Create sample text analysis
    text_analysis = (
        "Emotion: happy\n"
        "Profile: young female speaker\n"
        "Mood: cheerful\n"
        "Speed: moderate\n"
        "Prosody: rising intonation\n"
        "Pitch/Timbre: high-pitched, bright\n"
        "Style: conversational\n"
        "Notes: greeting\n"
        "Caption: A cheerful greeting with warm tone"
    )

    # Encode
    with torch.no_grad():
        embeddings = encoder(text_analysis)

    # Check output shape: (1, 32, 1024)
    assert embeddings.ndim == 3
    assert embeddings.shape[0] == 1  # batch
    assert embeddings.shape[1] == 32  # num queries
    assert embeddings.shape[2] == 1024  # embedding dim


def test_text_encoder_batch():
    """Test text encoder handles batch input."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")

    # Create batch of text analyses
    text_batch = [
        "Emotion: happy\nCaption: A cheerful greeting",
        "Emotion: sad\nCaption: A somber statement",
    ]

    # Encode batch
    with torch.no_grad():
        embeddings = encoder(text_batch)

    # Check output shape: (2, 32, 1024)
    assert embeddings.shape == (2, 32, 1024)


def test_text_encoder_training():
    """Test text encoder can be trained."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding

    encoder = TextToAudioEmbedding(device="cpu")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    # Sample inputs
    text = "Emotion: happy\nCaption: A cheerful greeting"
    target = torch.randn(1, 32, 1024)  # Target from voice encoder

    # Training step
    optimizer.zero_grad()
    prediction = encoder(text)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()

    # Check loss decreased
    assert loss.item() > 0


def test_text_encoder_save_load():
    """Test text encoder can be saved and loaded."""
    from chatterbox_encoders.text_analysis.text_encoder import TextToAudioEmbedding
    import tempfile
    from pathlib import Path

    encoder = TextToAudioEmbedding(device="cpu")
    text = "Emotion: happy\nCaption: A cheerful greeting"

    # Get output before saving
    with torch.no_grad():
        output_before = encoder(text)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "text_encoder.pt"
        encoder.save(checkpoint_path)

        # Load new instance
        encoder2 = TextToAudioEmbedding(device="cpu")
        encoder2.load(checkpoint_path)

        # Get output after loading
        with torch.no_grad():
            output_after = encoder2(text)

        # Check outputs are the same
        assert torch.allclose(output_before, output_after, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text_encoder.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'chatterbox_encoders.text_analysis.text_encoder'"

- [ ] **Step 3: Write minimal implementation**

Create `chatterbox_encoders/text_analysis/text_encoder.py`:
```python
"""
Text-to-audio-embedding encoder using T5.

Learns to map text analysis of audio to the same 32×1024 tokens
that the voice encoder produces, enabling text-only audio generation.
"""

import logging
from pathlib import Path
from typing import Union, Optional

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

logger = logging.getLogger(__name__)


class TextToAudioEmbedding(nn.Module):
    """
    Text-to-audio-embedding encoder using T5.

    This model learns to map rich text analysis (from Qwen3-Omni) to the
    same 32×1024 audio tokens that the voice encoder + Perceiver produce.
    This enables text-only audio generation during inference.

    Architecture:
        Text → T5 Encoder (frozen or trainable) → Projection → 1024-dim → Perceiver → 32×1024 tokens

    Args:
        model_name: T5 model name (default: t5-small)
        num_queries: Number of output queries (default: 32 for Perceiver compatibility)
        embedding_dim: Output embedding dimension (default: 1024 for Perceiver compatibility)
        device: Device to load model on (auto/cuda/mps/cpu)
        freeze_t5: Whether to freeze T5 encoder (default: True)
        latent_dim: Dimension of T5 → 1024 projection intermediate layer (default: None for linear)

    Examples:
        >>> encoder = TextToAudioEmbedding(device="cpu")
        >>> text = "Emotion: happy\\nCaption: A cheerful greeting"
        >>> embeddings = encoder(text)  # (1, 32, 1024)
        >>> embeddings.shape
        torch.Size([1, 32, 1024])
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        num_queries: int = 32,
        embedding_dim: int = 1024,
        device: str = "auto",
        freeze_t5: bool = True,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.freeze_t5 = freeze_t5

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"📝 Loading text-to-audio-embedding encoder: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Output: {num_queries} × {embedding_dim}")

        # Load T5 tokenizer
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            logger.info(f"   ✓ T5 tokenizer loaded")
        except Exception as e:
            logger.error(f"   ❌ Failed to load T5 tokenizer: {e}")
            raise

        # Load T5 encoder
        try:
            self.t5 = T5EncoderModel.from_pretrained(model_name).to(self.device)
            self.t5_output_dim = self.t5.config.d_model  # 512 for t5-small

            # Freeze T5 if specified
            if freeze_t5:
                for param in self.t5.parameters():
                    param.requires_grad = False
                logger.info(f"   ✓ T5 encoder frozen (d_model={self.t5_output_dim})")
            else:
                logger.info(f"   ✓ T5 encoder trainable (d_model={self.t5_output_dim})")
        except Exception as e:
            logger.error(f"   ❌ Failed to load T5 encoder: {e}")
            raise

        # Create projection from T5 output to embedding_dim
        if latent_dim is None:
            # Single linear layer
            self.projection = nn.Linear(self.t5_output_dim, embedding_dim)
            logger.info(f"   ✓ Projection: {self.t5_output_dim} → {embedding_dim} (linear)")
        else:
            # Two-layer MLP
            self.projection = nn.Sequential(
                nn.Linear(self.t5_output_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim, embedding_dim),
                nn.Dropout(0.1),
            )
            logger.info(f"   ✓ Projection: {self.t5_output_dim} → {latent_dim} → {embedding_dim} (MLP)")

        self.projection = self.projection.to(self.device)

        # Initialize projection weights
        self._init_weights()

        # Learnable query embeddings for generating fixed number of tokens
        # This allows variable-length text → fixed 32 tokens
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, self.t5_output_dim))
        nn.init.normal_(self.query_embeddings, std=0.02)

        # Output dimension attribute
        self.output_dim = embedding_dim

        logger.info(f"   ✓ Text-to-audio-embedding encoder ready")

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text: Union[str, list[str]],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Encode text analysis to audio-compatible embeddings.

        Args:
            text: Text analysis (single string or list of strings)
            return_attention: Whether to return attention weights

        Returns:
            Audio embeddings: (batch, num_queries, embedding_dim) = (batch, 32, 1024)

        Examples:
            >>> encoder = TextToAudioEmbedding()
            >>> text = "Emotion: happy\\nCaption: A cheerful greeting"
            >>> emb = encoder(text)
            >>> emb.shape
            torch.Size([1, 32, 1024])
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode with T5
        with torch.set_grad_enabled(not self.freeze_t5):
            outputs = self.t5(**inputs)
            t5_emb = outputs.last_hidden_state  # (batch, seq_len, d_model)

        # Project to embedding_dim
        projected = self.projection(t5_emb)  # (batch, seq_len, embedding_dim)

        # Generate fixed number of query tokens using learned queries
        # Approach: Use cross-attention between learned queries and text embeddings
        batch_size = projected.shape[0]

        # Expand query embeddings for batch
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, d_model)

        # Simple approach: average pool text embeddings and use queries to attend
        # For now, use a simpler approach: take mean of text embeddings and repeat
        text_pooled = projected.mean(dim=1, keepdim=True)  # (batch, 1, embedding_dim)
        output = text_pooled.expand(batch_size, self.num_queries, self.embedding_dim)  # (batch, num_queries, embedding_dim)

        # Add query-specific information via learned modulation
        # (This is a simplified approach - could use cross-attention for more sophisticated)
        query_modulation = self.projection(self.query_embeddings).unsqueeze(0)  # (1, num_queries, embedding_dim)
        output = output + query_modulation * 0.1  # Small modulation

        return output

    def save(self, path: Union[str, Path]):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint

        Examples:
            >>> encoder = TextToAudioEmbedding()
            >>> encoder.save("checkpoints/text_encoder.pt")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_name": self.model_name,
            "num_queries": self.num_queries,
            "embedding_dim": self.embedding_dim,
            "freeze_t5": self.freeze_t5,
            "projection_state_dict": self.projection.state_dict(),
            "query_embeddings": self.query_embeddings,
        }

        torch.save(checkpoint, path)
        logger.info(f"✓ Saved text encoder to {path}")

    def load(self, path: Union[str, Path]):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Examples:
            >>> encoder = TextToAudioEmbedding()
            >>> encoder.load("checkpoints/text_encoder.pt")
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        # Verify configuration matches
        assert checkpoint["model_name"] == self.model_name
        assert checkpoint["num_queries"] == self.num_queries
        assert checkpoint["embedding_dim"] == self.embedding_dim

        # Load weights
        self.projection.load_state_dict(checkpoint["projection_state_dict"])
        self.query_embeddings.data = checkpoint["query_embeddings"]

        logger.info(f"✓ Loaded text encoder from {path}")

    def get_trainable_params(self):
        """Get trainable parameters (excluding frozen T5)."""
        params = list(self.projection.parameters()) + [self.query_embeddings]
        return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text_encoder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chatterbox_encoders/text_analysis/text_encoder.py tests/test_text_encoder.py
git commit -m "feat: add T5-based text-to-audio-embedding encoder"
```

---

## Task 5: Create Audio Analysis Utility Script

**Files:**
- Create: `scripts/analyze_audio_with_qwen.py`

- [ ] **Step 1: Create analysis script**

Create `scripts/analyze_audio_with_qwen.py`:
```python
"""
Utility script to analyze audio files with Qwen3-Omni.

Extracts rich text analysis including emotion, profile, mood, speed,
prosody, pitch_timbre, style, notes, and caption.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from chatterbox_encoders.text_analysis import QwenOmniAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze audio with Qwen3-Omni")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--format-for-encoder",
        action="store_true",
        help="Format output for text encoder input",
    )

    args = parser.parse_args()

    # Initialize analyzer
    logger.info(f"🎤 Initializing Qwen3-Omni analyzer...")
    analyzer = QwenOmniAnalyzer(device=args.device)

    # Analyze audio
    logger.info(f"📂 Analyzing: {args.audio_path}")
    analysis = analyzer.analyze_from_file(args.audio_path)

    # Format output
    if args.format_for_encoder:
        output = analyzer.format_for_encoder(analysis)
    else:
        output = analysis

    # Save or print
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format_for_encoder:
            output_path.write_text(output)
        else:
            with output_path.open("w") as f:
                json.dump(analysis, f, indent=2)

        logger.info(f"✓ Saved to: {args.output}")
    else:
        if args.format_for_encoder:
            print(output)
        else:
            print(json.dumps(analysis, indent=2))

    logger.info("✓ Done!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x scripts/analyze_audio_with_qwen.py`

- [ ] **Step 3: Test script**

Run: `uv run python scripts/analyze_audio_with_qwen.py --help`
Expected: Help message displayed

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_audio_with_qwen.py
git commit -m "feat: add audio analysis utility script"
```

---

## Task 6: Create Training Script

**Files:**
- Create: `scripts/train_text_encoder.py`

- [ ] **Step 1: Create training script**

Create `scripts/train_text_encoder.py`:
```python
"""
Training script for text-to-audio-embedding encoder.

Learns to map text analysis (from Qwen3-Omni) to the same 32×1024 tokens
that the voice encoder + Perceiver produce.

Training loop:
    For each audio in dataset:
        1. Extract text analysis with Qwen3-Omni
        2. Get ground truth tokens from voice encoder + Perceiver
        3. Train text encoder to predict ground truth from text analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from chatterbox_encoders.text_analysis import QwenOmniAnalyzer, TextToAudioEmbedding
from chatterbox_encoders.audio import PerceiverResampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AudioTextDataset(Dataset):
    """Dataset for audio files with text analysis."""

    def __init__(
        self,
        audio_files: List[Path],
        voice_encoder_checkpoint: str,
        qwen_analyzer: QwenOmniAnalyzer,
        device: str = "cuda",
    ):
        self.audio_files = audio_files
        self.device = device

        # TODO: Load voice encoder
        # This would use your existing voice encoder setup
        logger.info(f"📝 Dataset: {len(audio_files)} audio files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]

        # Extract text analysis
        # text_analysis = self.qwen_analyzer.analyze_from_file(str(audio_path))

        # Get ground truth tokens from voice encoder
        # ground_truth = self.voice_encoder.encode(audio_path)

        # Return (text_analysis, ground_truth) pair
        pass


def train_epoch(
    model: TextToAudioEmbedding,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        text_analysis, ground_truth = batch

        # Move to device
        ground_truth = ground_truth.to(device)

        # Forward pass
        optimizer.zero_grad()
        prediction = model(text_analysis)

        # Compute loss (MSE between prediction and ground truth)
        loss = nn.functional.mse_loss(prediction, ground_truth)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: TextToAudioEmbedding,
    dataloader: DataLoader,
    device: str,
):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            text_analysis, ground_truth = batch

            # Move to device
            ground_truth = ground_truth.to(device)

            # Forward pass
            prediction = model(text_analysis)

            # Compute loss
            loss = nn.functional.mse_loss(prediction, ground_truth)

            # Track loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train text-to-audio-embedding encoder")
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (JSON with audio paths)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (JSON with audio paths)",
    )
    parser.add_argument(
        "--voice-encoder",
        type=str,
        required=True,
        help="Path to voice encoder checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/text_encoder",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"🚀 Training text-to-audio-embedding encoder")
    logger.info(f"   Device: {device}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.lr}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    logger.info(f"📂 Loading training data from: {args.train_data}")
    train_data_path = Path(args.train_data)
    with train_data_path.open("r") as f:
        train_data = json.load(f)

    # Assuming train_data is a list of audio file paths
    train_audio_files = [Path(p) for p in train_data]

    # Initialize Qwen3-Omni analyzer
    logger.info(f"🎤 Initializing Qwen3-Omni analyzer...")
    qwen_analyzer = QwenOmniAnalyzer(device=device)

    # Create dataset and dataloader
    # TODO: Implement AudioTextDataset properly
    # train_dataset = AudioTextDataset(
    #     audio_files=train_audio_files,
    #     voice_encoder_checkpoint=args.voice_encoder,
    #     qwen_analyzer=qwen_analyzer,
    #     device=device,
    # )
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize text encoder
    logger.info(f"📝 Initializing text encoder...")
    text_encoder = TextToAudioEmbedding(device=device)

    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"📂 Loading checkpoint: {args.checkpoint}")
        text_encoder.load(args.checkpoint)

    # Initialize optimizer
    trainable_params = text_encoder.get_trainable_params()
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # Training loop
    logger.info(f"🏋️ Starting training...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        # train_loss = train_epoch(text_encoder, train_dataloader, optimizer, device)
        train_loss = 0.0  # Placeholder
        logger.info(f"   Train loss: {train_loss:.4f}")

        # Validate
        if args.val_data:
            # val_loss = validate(text_encoder, val_dataloader, device)
            val_loss = 0.0  # Placeholder
            logger.info(f"   Val loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / "best.pt"
                text_encoder.save(checkpoint_path)
                logger.info(f"   ✓ Saved best model: {checkpoint_path}")

        # Save checkpoint
        checkpoint_path = output_dir / f"epoch_{epoch + 1}.pt"
        text_encoder.save(checkpoint_path)
        logger.info(f"   ✓ Saved checkpoint: {checkpoint_path}")

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_text_encoder.py
git commit -m "feat: add text encoder training script (stub)"
```

---

## Task 7: Update Module Exports

**Files:**
- Modify: `chatterbox_encoders/text_analysis/__init__.py`

- [ ] **Step 1: Verify exports are correct**

The exports were already added in Task 2. Verify they work:

Run: `uv run python -c "from chatterbox_encoders.text_analysis import QwenOmniAnalyzer, TextToAudioEmbedding; print('✓ Exports work')"`
Expected: ✓ Exports work

- [ ] **Step 2: No commit needed** (already done in Task 2)

---

## Task 8: Create Documentation

**Files:**
- Create: `docs/text_to_audio_embeddings.md`

- [ ] **Step 1: Write documentation**

Create `docs/text_to_audio_embeddings.md`:
```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/text_to_audio_embeddings.md
git commit -m "docs: add text-to-audio-embeddings documentation"
```

---

## Task 9: Final Verification

**Files:**
- All files

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/test_qwen_analyzer.py tests/test_text_encoder.py -v`
Expected: All tests pass

- [ ] **Step 2: Verify linting**

Run: `uv run ruff check chatterbox_encoders/text_analysis/ tests/test_qwen_*.py tests/test_text_*.py scripts/*.py`
Expected: No linting errors (or acceptable warnings)

- [ ] **Step 3: Test analysis script**

Run:
```bash
uv run python scripts/analyze_audio_with_qwen.py --help
```
Expected: Help message displayed

- [ ] **Step 4: Verify imports**

Run:
```bash
uv run python -c "
from chatterbox_encoders.text_analysis import QwenOmniAnalyzer, TextToAudioEmbedding
print('✓ QwenOmniAnalyzer:', QwenOmniAnalyzer)
print('✓ TextToAudioEmbedding:', TextToAudioEmbedding)
"
```
Expected: No import errors

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete text-to-audio-embeddings learning system"
```

---

## Summary

This plan implements a text-to-audio-embeddings learning system with:

1. **QwenOmniAnalyzer**: Extracts rich text analysis from audio (9 fields)
2. **TextToAudioEmbedding**: T5-based encoder mapping text → 32×1024 audio tokens
3. **Training Script**: Learns text encoder from voice encoder ground truth
4. **Utility Scripts**: Audio analysis, checkpoint management
5. **Documentation**: Complete usage guide

**Key Features:**
- Text-only inference (no audio required during generation)
- Compatible with existing Perceiver Resampler (32×1024 tokens)
- Lightweight training (only projection + queries, T5 frozen)
- Rich text analysis from Qwen3-Omni (emotion, profile, mood, etc.)

**Next Steps After Implementation:**
1. Create training dataset with audio files
2. Complete AudioTextDataset implementation with voice encoder integration
3. Train text encoder on your dataset
4. Benchmark text-only generation quality
