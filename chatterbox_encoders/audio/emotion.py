"""
Emotion projection module.

Projects scalar emotion values to model dimension (1024).
"""

import torch
import torch.nn as nn
import logging
from typing import Union

from chatterbox_encoders.config.defaults import EMOTION_PROJ_INPUT_DIM, EMOTION_PROJ_OUTPUT_DIM

logger = logging.getLogger(__name__)


class EmotionProjector(nn.Module):
    """
    Project scalar emotion value to model dimension.

    This converts an emotion scalar (0.0 to 1.0) into a 1024-dimensional
    embedding that serves as a conditioning token in the T3 model.

    Higher emotion values lead to more expressive speech output.

    Args:
        input_dim: Input dimension (default: 1)
        output_dim: Output dimension (default: 1024)

    Examples:
        >>> projector = EmotionProjector()
        >>> emotion = torch.tensor([[[0.7]]])  # 70% expressiveness
        >>> emotion_token = projector(emotion)  # (1, 1, 1024) - becomes 1 token
    """

    def __init__(
        self,
        input_dim: int = EMOTION_PROJ_INPUT_DIM,
        output_dim: int = EMOTION_PROJ_OUTPUT_DIM,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # No bias for emotion projection
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

        logger.info(f"EmotionProjector: {input_dim} → {output_dim}")

    def forward(
        self,
        emotion_value: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Project emotion value to model dimension.

        Args:
            emotion_value: Emotion value
                - Can be: float, or
                - Tensor of shape: (B, 1, 1) or (1, 1) or (1,)
                - Range: 0.0 (neutral) to 1.0 (maximum expression)

        Returns:
            torch.Tensor: Emotion embedding
                - Shape: (B, 1, 1024)
                - Ready to be used as 1 conditioning token

        Examples:
            >>> projector = EmotionProjector()
            >>> emotion = 0.7  # 70% expressiveness
            >>> emotion_token = projector(emotion)
            >>> emotion_token.shape
            torch.Size([1, 1, 1024])
        """
        # Convert scalar to tensor if needed
        if isinstance(emotion_value, (int, float)):
            emotion_value = torch.tensor([[[emotion_value]]])
        elif isinstance(emotion_value, torch.Tensor):
            # Ensure shape (B, 1, 1)
            if emotion_value.dim() == 0:
                emotion_value = emotion_value.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif emotion_value.dim() == 1:
                emotion_value = emotion_value.unsqueeze(0).unsqueeze(0)
            elif emotion_value.dim() == 2:
                emotion_value = emotion_value.unsqueeze(0)

        projected = self.projection(emotion_value)

        return projected
