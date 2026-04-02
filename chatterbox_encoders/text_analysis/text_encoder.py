"""
Text-to-audio-embedding encoder using T5.

Learns to map text analysis of audio to the same 32×1024 tokens
that the voice encoder produces, enabling text-only audio generation.
"""

import logging
from pathlib import Path
from typing import Optional, Union

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
            logger.info("   ✓ T5 tokenizer loaded")
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
