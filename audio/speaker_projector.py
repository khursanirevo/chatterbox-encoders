"""
Speaker embedding projection module.

Projects 256-dimensional speaker embeddings to model dimension (1024).
"""

import torch
import torch.nn as nn
import logging
from typing import Union

from chatterbox_encoders.config.defaults import SPEAKER_PROJ_INPUT_DIM, SPEAKER_PROJ_OUTPUT_DIM

logger = logging.getLogger(__name__)


class SpeakerProjector(nn.Module):
    """
    Project 256-dimensional speaker embedding to model dimension.

    This is a simple linear projection that converts the speaker embedding
    from VoiceEncoder (256-dim) to the model dimension (1024-dim for LLaMA).

    The projected embedding becomes a single conditioning token in the T3 model.

    Args:
        input_dim: Input dimension (default: 256)
        output_dim: Output dimension (default: 1024)

    Examples:
        >>> projector = SpeakerProjector()
        >>> spk_emb = torch.randn(1, 256)  # From VoiceEncoder
        >>> spk_token = projector(spk_emb)  # (1, 1024) - becomes 1 token
    """

    def __init__(
        self,
        input_dim: int = SPEAKER_PROJ_INPUT_DIM,
        output_dim: int = SPEAKER_PROJ_OUTPUT_DIM,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.projection = nn.Linear(input_dim, output_dim)

        logger.info(f"SpeakerProjector: {input_dim} → {output_dim}")

    def forward(
        self,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project speaker embedding to model dimension.

        Args:
            speaker_embedding: Speaker embedding tensor
                - Shape: (B, 256) or (256,)
                - From VoiceEncoder

        Returns:
            torch.Tensor: Projected embedding
                - Shape: (B, 1024)
                - Ready to be used as 1 conditioning token

        Examples:
            >>> projector = SpeakerProjector()
            >>> spk_emb = torch.randn(2, 256)
            >>> projected = projector(spk_emb)
            >>> projected.shape
            torch.Size([2, 1024])
        """
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        projected = self.projection(speaker_embedding)

        return projected
