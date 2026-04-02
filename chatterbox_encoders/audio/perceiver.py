"""
Perceiver Resampler module for token compression.

Compresses variable-length speech token embeddings to fixed 32 tokens
using cross-attention and self-attention mechanisms.
"""

import math
import logging
from typing import Optional

import torch
from torch import nn
from einops import rearrange

from chatterbox_encoders.config.constants import (
    MODEL_DIM,
    PERCEIVER_QUERIES,
    PERCEIVER_QUERY_DIM,
    PERCEIVER_NUM_HEADS,
)
from chatterbox_encoders.config.defaults import (
    PERCEIVER_NUM_QUERIES,
    PERCEIVER_QUERY_DIM,
    PERCEIVER_EMBEDDING_DIM,
    PERCEIVER_NUM_HEADS,
    PERCEIVER_DROPOUT,
)

logger = logging.getLogger(__name__)


class RelativePositionBias(nn.Module):
    """Relative position bias for attention."""

    def __init__(
        self,
        scale: float,
        causal: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        heads: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        causal: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        """Bucket relative position."""
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)

        return ret

    def forward(self, qk_dots: torch.Tensor) -> torch.Tensor:
        """Add relative position bias to attention scores."""
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return qk_dots + (bias * self.scale)


class AttentionQKV(nn.Module):
    """Multi-head attention with Q, K, V projections."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        dropout_rate: float = 0.1,
        scale: Optional[float] = None,
        flash: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.flash = flash
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention."""
        q, k, v = [self._split_heads(tensor) for tensor in [q, k, v]]

        if self.flash:
            out = self._flash_attention(q, k, v, mask=mask)
        else:
            out = self._scaled_dot_product_attention(q, k, v, mask=mask)

        out = self._combine_heads(out)

        return out

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split heads."""
        bs, length, _ = x.shape
        x = x.view(bs, length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine heads."""
        bs, _, length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bs, length, -1)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scaled dot-product attention."""
        sim = torch.einsum("bhlt,bhls->bhts", q, k) * self.scale
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        return torch.einsum("bhts,bhls->bhlt", attn, v)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flash attention."""
        config = {
            "enable_flash": True,
            "enable_math": True,
            "enable_mem_efficient": True,
        }
        with torch.backends.cuda.sdp_kernel(**config):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )
        return out


class AttentionBlock(nn.Module):
    """Attention block with separate Q, K, V projections."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        relative_pos_embeddings: bool = False,
        flash_attention: bool = True,
        dropout_rate: float = 0.2,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKV(
            self.num_heads,
            channels // self.num_heads,
            dropout_rate=dropout_rate,
            flash=flash_attention,
            scale=scale,
        )

        self.proj_out = nn.Linear(channels, channels)

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(channels // self.num_heads) ** 0.5,
                causal=False,
                heads=num_heads,
                num_buckets=32,
                max_distance=64,
            )
        else:
            self.relative_pos_embeddings = None

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention block."""
        b1, c1, *spatial1 = x1.shape
        b2, c2, *spatial2 = x2.shape

        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return (x1 + h).reshape(b1, c1, *spatial1)


class Perceiver(nn.Module):
    """
    Perceiver resampler for compressing variable-length sequences.

    Uses learnable query tokens and cross-attention to compress
    variable-length speech token embeddings to fixed 32 tokens.

    Args:
        pre_attention_query_token: Number of output query tokens (default: 32)
        pre_attention_query_size: Dimension of query tokens (default: 1024)
        embedding_dim: Model embedding dimension (default: 1024)
        num_attn_heads: Number of attention heads (default: 4)

    Examples:
        >>> perceiver = PerceiverResampler()
        >>> speech_emb = torch.randn(1, 150, 1024)  # Variable length
        >>> compressed = perceiver(speech_emb)
        >>> compressed.shape
        torch.Size([1, 32, 1024])
    """

    def __init__(
        self,
        pre_attention_query_token: int = PERCEIVER_NUM_QUERIES,
        pre_attention_query_size: int = PERCEIVER_QUERY_DIM,
        embedding_dim: int = PERCEIVER_EMBEDDING_DIM,
        num_attn_heads: int = PERCEIVER_NUM_HEADS,
    ):
        super().__init__()

        # Initialize learnable query tokens
        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, pre_attention_query_token, pre_attention_query_size)
        )

        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (
            pre_attention_query_token + pre_attention_query_token
        ))

        self.pre_attention_query.data.uniform_(-query_variance, query_variance)

        # Attention block
        self.attn = AttentionBlock(
            channels=embedding_dim,
            num_heads=num_attn_heads,
            relative_pos_embeddings=False,
            flash_attention=True,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Perceiver.

        Args:
            h: Input tensor (B, T_in, 1024) speech token embeddings

        Returns:
            torch.Tensor: Compressed output (B, 32, 1024)

        Examples:
            >>> perceiver = Perceiver()
            >>> speech_emb = torch.randn(1, 150, 1024)
            >>> compressed = perceiver(speech_emb)
            >>> compressed.shape
            torch.Size([1, 32, 1024])
        """
        # Expand queries to batch size
        query_ = self.pre_attention_query.expand(h.shape[0], -1, -1)

        # Cross-attention: queries attend to input
        pre_att = self.attn(query_, h)

        # Self-attention: queries attend to themselves
        attn = self.attn(pre_att, pre_att)

        return attn


class PerceiverResampler(nn.Module):
    """
    Wrapper for Perceiver with clearer API.

    This is the main class to use for compressing speech token embeddings.

    Args:
        num_queries: Number of output tokens (default: 32)
        query_dim: Query token dimension (default: 1024)
        embedding_dim: Input embedding dimension (default: 1024)
        num_heads: Number of attention heads (default: 4)

    Examples:
        >>> resampler = PerceiverResampler(num_queries=32)
        >>> speech_emb = torch.randn(1, 150, 1024)
        >>> compressed = resampler(speech_emb)
        >>> compressed.shape
        torch.Size([1, 32, 1024])
    """

    def __init__(
        self,
        num_queries: int = PERCEIVER_NUM_QUERIES,
        query_dim: int = PERCEIVER_QUERY_DIM,
        embedding_dim: int = PERCEIVER_EMBEDDING_DIM,
        num_heads: int = PERCEIVER_NUM_HEADS,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.query_dim = query_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.perceiver = Perceiver(
            pre_attention_query_token=num_queries,
            pre_attention_query_size=query_dim,
            embedding_dim=embedding_dim,
            num_attn_heads=num_heads,
        )

        logger.info(
            f"PerceiverResampler: {embedding_dim} → {num_queries} × {embedding_dim}"
        )

    def forward(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compress speech token embeddings.

        Args:
            speech_embeddings: Speech token embeddings
                - Shape: (B, T_in, 1024)
                - T_in: Variable length (e.g., 150 for 6 seconds @ 25 tok/sec)

        Returns:
            torch.Tensor: Compressed embeddings
                - Shape: (B, 32, 1024)
                - Fixed 32 tokens regardless of input length

        Examples:
            >>> resampler = PerceiverResampler()
            >>> emb = torch.randn(1, 150, 1024)  # 150 tokens
            >>> compressed = resampler(emb)
            >>> compressed.shape
            torch.Size([1, 32, 1024])
        """
        return self.perceiver(speech_embeddings)
