"""
Pipeline components for Chatterbox encoding.

This module provides high-level pipelines that combine multiple components:
- AudioPromptBuilder: Create complete audio prompts from reference audio
- T3ConditioningBuilder: Build T3 conditioning for text-to-speech
- S3GenReferenceBuilder: Build S3Gen reference embeddings
"""

__all__ = [
    "AudioPromptBuilder",
    "T3ConditioningBuilder",
    "S3GenReferenceBuilder",
]
