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
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextToAudioEmbedding(nn.Module):
    """
    Text-to-audio-embedding encoder using Sentence Transformers with Cross-Attention.

    This model learns to map rich text analysis (from Qwen3-Omni) to the
    same 32×1024 audio tokens that the voice encoder + Perceiver produce.
    This enables text-only audio generation during inference.

    Architecture:
        Text → Sentence Transformer (frozen, 768-dim) → Projection (768→1024)
                                                        ↓
                                    Learnable Query Embeddings (32, 768)
                                                        ↓
                                  Cross-Attention (Query ← Text)
                                                        ↓
                                      32 unique tokens (each attends differently)

    Args:
        model_name: Sentence Transformer model name (default: sentence-transformers/all-mpnet-base-v2)
        num_queries: Number of output queries (default: 32 for Perceiver compatibility)
        embedding_dim: Output embedding dimension (default: 1024 for Perceiver compatibility)
        num_heads: Number of attention heads (default: 8)
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
        num_heads: int = 8,
        device: str = "auto",
        freeze_encoder: bool = True,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.freeze_encoder = freeze_encoder

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"📝 Loading text-to-audio-embedding encoder: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Output: {num_queries} × {embedding_dim}")
        logger.info(f"   Architecture: Cross-Attention with {num_heads} heads")

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

        # Learnable query embeddings for cross-attention
        # These learnable queries will attend to the text representation
        self.query_embeddings = nn.Parameter(
            torch.randn(num_queries, self.encoder_output_dim) * 0.02
        )
        logger.info(f"   ✓ Learnable query embeddings: ({num_queries}, {self.encoder_output_dim})")

        # Cross-attention layer: queries attend to text representation
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        logger.info(f"   ✓ Cross-attention: {num_heads} heads, dim={embedding_dim}")

        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(embedding_dim)
        logger.info("   ✓ Layer normalization for residual connection")

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
        Encode text analysis to audio-compatible embeddings using cross-attention.

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
            text_embeddings = self.sentence_transformer.encode(
                text,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=False,
            )  # (batch, encoder_output_dim)

        # Ensure embeddings are on correct device
        text_embeddings = text_embeddings.to(self.device)  # (batch, encoder_output_dim)

        # Project text to embedding_dim
        batch_size = text_embeddings.shape[0]
        text_projected = self.projection(text_embeddings)  # (batch, embedding_dim)

        # Expand text to (batch, 1, embedding_dim) for use as key/value in attention
        text_kv = text_projected.unsqueeze(1)  # (batch, 1, embedding_dim)

        # Project learnable queries to embedding_dim
        queries_projected = self.projection(self.query_embeddings)  # (num_queries, embedding_dim)

        # Expand queries to batch dimension
        queries = queries_projected.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, embedding_dim)

        # Apply cross-attention: queries attend to text representation
        # query: (batch, num_queries, embedding_dim)
        # key/value: (batch, 1, embedding_dim)
        attn_output, _ = self.cross_attention(
            query=queries,
            key=text_kv,
            value=text_kv,
        )  # (batch, num_queries, embedding_dim)

        # Residual connection + layer norm
        output = self.layer_norm(queries + attn_output)  # (batch, num_queries, embedding_dim)

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
            "num_heads": self.num_heads,
            "freeze_encoder": self.freeze_encoder,
            "projection_state_dict": self.projection.state_dict(),
            "query_embeddings": self.query_embeddings,
            "cross_attention_state_dict": self.cross_attention.state_dict(),
            "layer_norm_state_dict": self.layer_norm.state_dict(),
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
        assert checkpoint["num_heads"] == self.num_heads

        # Load weights
        self.projection.load_state_dict(checkpoint["projection_state_dict"])
        self.query_embeddings.data = checkpoint["query_embeddings"]
        self.cross_attention.load_state_dict(checkpoint["cross_attention_state_dict"])
        self.layer_norm.load_state_dict(checkpoint["layer_norm_state_dict"])

        logger.info(f"✓ Loaded text encoder from {path}")

    def get_trainable_params(self):
        """Get trainable parameters (excluding frozen encoder)."""
        params = (
            list(self.projection.parameters()) +
            [self.query_embeddings] +
            list(self.cross_attention.parameters()) +
            list(self.layer_norm.parameters())
        )
        return params


def multi_scale_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    num_scales: int = 4,
) -> torch.Tensor:
    """
    Compute MSE loss at multiple scales.

    This loss function divides the tokens into chunks and computes MSE loss
    for each chunk separately, then averages the losses. This encourages
    the model to learn fine-grained token-level representations.

    Args:
        prediction: Predicted tokens (batch, num_queries, embedding_dim)
        ground_truth: Ground truth tokens (batch, num_queries, embedding_dim)
        num_scales: Number of chunks to divide tokens into (default: 4)

    Returns:
        Average loss across all scales

    Examples:
        >>> pred = torch.randn(2, 32, 1024)
        >>> gt = torch.randn(2, 32, 1024)
        >>> loss = multi_scale_loss(pred, gt, num_scales=4)
        >>> loss.item() > 0
        True
    """
    batch_size, num_queries, embedding_dim = prediction.shape
    chunk_size = num_queries // num_scales

    total_loss = 0.0

    for i in range(num_scales):
        start_idx = i * chunk_size
        # Ensure last chunk includes all remaining tokens
        end_idx = start_idx + chunk_size if i < num_scales - 1 else num_queries

        chunk_pred = prediction[:, start_idx:end_idx, :]
        chunk_gt = ground_truth[:, start_idx:end_idx, :]

        total_loss += F.mse_loss(chunk_pred, chunk_gt)

    return total_loss / num_scales
