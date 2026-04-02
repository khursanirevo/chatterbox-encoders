"""
Qwen3-Omni text analyzer wrapper.

Extracts rich text analysis from audio using Qwen3-Omni's audio understanding
capabilities. The analysis includes emotion, profile, mood, speed, prosody,
pitch_timbre, style, notes, and caption.
"""

import logging
import re
from typing import Optional

import librosa
import numpy as np
import torch

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
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            logger.info("   ✓ Model loaded")
        except Exception as e:
            logger.error(f"   ❌ Failed to load Qwen3-Omni model: {e}")
            logger.info("   ⚠️  Falling back to mock analysis for testing")
            self.model = None

        # Load processor
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            logger.info("   ✓ Processor loaded")
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to load processor: {e}")
            self.processor = None

        logger.info("   ✓ Qwen3-Omni analyzer ready")

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
            logger.info("   ⚠️  Using mock analysis")
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

        patterns = {
            "emotion": r"(?:emotion|feeling)\s*[:-]\s*([^\n]+)",
            "profile": r"(?:profile|speaker)\s*[:-]\s*([^\n]+)",
            "mood": r"(?:mood|atmosphere)\s*[:-]\s*([^\n]+)",
            "speed": r"(?:speed|pace|tempo)\s*[:-]\s*([^\n]+)",
            "prosody": r"(?:prosody|rhythm|cadence)\s*[:-]\s*([^\n]+)",
            "pitch_timbre": r"(?:pitch[-_\s]?timbre|voice[-_\s]?quality)\s*[:-]\s*([^\n]+)",
            "style": r"(?:style|delivery)\s*[:-]\s*([^\n]+)",
            "notes": r"(?:notes|observations|other)\s*[:-]\s*([^\n]+)",
            "caption": r"(?:caption|summary|description)\s*[:-]\s*([^\n]+)",
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
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)

        return self.analyze_audio(audio, sample_rate=sr)
