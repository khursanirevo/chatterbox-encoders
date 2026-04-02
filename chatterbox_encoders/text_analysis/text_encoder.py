"""
Text-to-audio-embedding encoder using Sentence Transformers.

Learns to map text analysis of audio to the same 32×1024 tokens
that the voice encoder produces, enabling text-only audio generation.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextToAudioEmbedding(nn.Module):
    """
    Text-to-audio-embedding encoder using Sentence Transformers.

    This model learns to map rich text analysis (from Qwen3-Omni) to the
    same 32×1024 audio tokens that the voice encoder + Perceiver produce.
    This enables text-only audio generation during inference.

    Architecture:
        Text → Sentence Transformer (frozen) → Projection → 1024-dim → 32 tokens

    Args:
        model_name: Sentence Transformer model name (default: sentence-transformers/all-mpnet-base-v2)
        num_queries: Number of output queries (default: 32 for Perceiver compatibility)
        embedding_dim: Output embedding dimension (default: 1024 for Perceiver compatibility)
        device: Device to load model on (auto/cuda/mps/cpu)
        freeze_encoder: Whether to freeze sentence transformer (default: True)
        latent_dim: Dimension of projection intermediate layer (default: None for linear)

    Examples:
        >>> encoder = TextToAudioEmbedding(device="cpu")
        >>> text = "Emotion: happy\\nCaption: A cheerful greeting"
        >>> embeddings = encoder(text)  # (1, 32, 1024)
        >>> embeddings.shape
        torch.Size([1, 32, 1024])
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        num_queries: int = 32,
        embedding_dim: int = 1024,
        device: str = "auto",
        freeze_encoder: bool = True,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.freeze_encoder = freeze_encoder

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"📝 Loading text-to-audio-embedding encoder: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Output: {num_queries} × {embedding_dim}")

        # Load Sentence Transformer
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            self.encoder_output_dim = self.sentence_transformer.get_sentence_embedding_dimension()
            logger.info(f"   ✓ Sentence Transformer loaded (dim={self.encoder_output_dim})")

            # Move to device
            self.sentence_transformer = self.sentence_transformer.to(self.device)

            # Freeze encoder if specified
            if freeze_encoder:
                for param in self.sentence_transformer.parameters():
                    param.requires_grad = False
                logger.info("   ✓ Sentence Transformer frozen")
            else:
                logger.info("   ✓ Sentence Transformer trainable")
        except Exception as e:
            logger.error(f"   ❌ Failed to load Sentence Transformer: {e}")
            raise

        # Create projection from encoder output to embedding_dim
        if latent_dim is None:
            # Single linear layer
            self.projection = nn.Linear(self.encoder_output_dim, embedding_dim)
            logger.info(f"   ✓ Projection: {self.encoder_output_dim} → {embedding_dim} (linear)")
        else:
            # Two-layer MLP
            self.projection = nn.Sequential(
                nn.Linear(self.encoder_output_dim, latent_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim, embedding_dim),
                nn.Dropout(0.1),
            )
            logger.info(f"   ✓ Projection: {self.encoder_output_dim} → {latent_dim} → {embedding_dim} (MLP)")

        self.projection = self.projection.to(self.device)

        # Initialize projection weights
        self._init_weights()

        # Learnable query embeddings for generating fixed number of tokens
        # This allows variable-length text → fixed 32 tokens
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, self.encoder_output_dim))
        nn.init.normal_(self.query_embeddings, std=0.02)

        # Output dimension attribute
        self.output_dim = embedding_dim

        logger.info("   ✓ Text-to-audio-embedding encoder ready")

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
    ) -> torch.Tensor:
        """
        Encode text analysis to audio-compatible embeddings.

        Args:
            text: Text analysis (single string or list of strings)

        Returns:
            Audio embeddings: (batch, num_queries, embedding_dim) = (batch, 32, 1024)

        Examples:
            >>> encoder = TextToAudioEmbedding()
            >>> text = "Emotion: happy\\nCaption: A cheerful greeting"
            >>> emb = encoder(text)
            >>> emb.shape
            torch.Size([1, 32, 1024])
        """
        # Normalize input to list
        if isinstance(text, str):
            text = [text]

        # Encode with Sentence Transformer
        with torch.set_grad_enabled(not self.freeze_encoder):
            # SentenceTransformer.encode returns numpy array by default
            # We need to use it differently to get tensors with gradients
            embeddings = self.sentence_transformer.encode(
                text,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=False,
            )  # (batch, encoder_output_dim)

        # Ensure embeddings are on correct device
        embeddings = embeddings.to(self.device)

        # Project to embedding_dim
        projected = self.projection(embeddings)  # (batch, embedding_dim)

        # Generate fixed number of query tokens using learned queries
        batch_size = projected.shape[0]

        # Expand to (batch, 1, embedding_dim)
        text_expanded = projected.unsqueeze(1)  # (batch, 1, embedding_dim)

        # Expand to (batch, num_queries, embedding_dim)
        output = text_expanded.expand(batch_size, self.num_queries, self.embedding_dim)

        # Add query-specific information via learned modulation
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
            "freeze_encoder": self.freeze_encoder,
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
        """Get trainable parameters (excluding frozen encoder)."""
        params = list(self.projection.parameters()) + [self.query_embeddings]
        return params
