"""
Training script for TextToAudioEmbedding (T5-based).

Learns to map text labels to the same 32×1024 audio tokens that the voice encoder + Perceiver produce.

Training loop:
    For each (audio, text_label) pair:
        1. Extract ground truth: Audio → S3Tokenizer → Embedding → Perceiver → 32×1024 tokens
        2. Get prediction: Text label → T5 → Projection → 32×1024 tokens
        3. Train with MSE loss

Usage:
    python scripts/train_text_encoder.py \
        --data-dir data/audio_text_pairs/ \
        --output-dir checkpoints/text_encoder \
        --epochs 10 \
        --batch-size 4

Expected data-dir structure:
    data_dir/
        audio_001.wav
        audio_002.wav
        ...
        labels.json  # {"audio_001.wav": "text label", ...}
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chatterbox_encoders.audio import PerceiverResampler, S3Tokenizer
from chatterbox_encoders.text_analysis import TextToAudioEmbedding
from chatterbox_encoders.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpeechTokenEmbedding(nn.Module):
    """Learnable embedding layer for speech tokens."""
    def __init__(self, vocab_size: int = 6561, embedding_dim: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)


class AudioTextDataset(Dataset):
    """
    Dataset for audio + text label pairs.

    Each item returns:
        - text_label: str (the text description/caption)
        - ground_truth: torch.Tensor (1, 32, 1024) from voice encoder + Perceiver
    """

    def __init__(
        self,
        data_dir: Path,
        s3_tokenizer: S3Tokenizer,
        speech_embedding: nn.Module,
        perceiver: PerceiverResampler,
        device: str,
        max_duration: float = 30.0,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing audio files and JSON labels
            s3_tokenizer: S3Tokenizer for speech tokenization
            speech_embedding: SpeechTokenEmbedding for embedding tokens
            perceiver: PerceiverResampler for compression
            device: Device to use
            max_duration: Maximum audio duration in seconds

        Expected data_dir structure:
            data_dir/
                audio_001.wav
                audio_002.wav
                ...
                labels.json
        """
        self.data_dir = Path(data_dir)
        self.device = device
        self.max_duration = max_duration

        # Store models
        self.s3_tokenizer = s3_tokenizer
        self.speech_embedding = speech_embedding
        self.perceiver = perceiver

        # Load labels
        labels_file = self.data_dir / "labels.json"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        with labels_file.open("r") as f:
            self.labels = json.load(f)

        # Verify all audio files exist
        self.audio_files = []
        for audio_name in self.labels.keys():
            audio_path = self.data_dir / audio_name
            if audio_path.exists():
                self.audio_files.append(audio_path)
            else:
                logger.warning(f"Audio file not found: {audio_path}")

        logger.info(f"📁 Dataset: {len(self.audio_files)} audio files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            text_label: str (the text description)
            ground_truth: torch.Tensor (32, 1024) from voice encoder
        """
        audio_path = self.audio_files[idx]
        audio_name = audio_path.name

        # Get text label
        text_label = self.labels[audio_name]

        # Load audio at 16kHz
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        audio = audio.astype(np.float32)

        # Trim to max duration
        max_samples = int(self.max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract ground truth tokens from audio
        with torch.no_grad():
            # Tokenize
            tokens, _ = self.s3_tokenizer.forward([audio])  # (1, T)

            # Convert to embeddings
            embeddings = self.speech_embedding(tokens)  # (1, T, 1024)

            # Compress with Perceiver
            compressed = self.perceiver(embeddings)  # (1, 32, 1024)

        # Remove batch dimension for dataset
        ground_truth = compressed.squeeze(0)  # (32, 1024)

        return text_label, ground_truth


def collate_fn(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of (text_label, ground_truth) tuples

    Returns:
        text_labels: List of strings
        ground_truth: torch.Tensor (batch, 32, 1024)
    """
    text_labels = [item[0] for item in batch]
    ground_truth = torch.stack([item[1] for item in batch])
    return text_labels, ground_truth


def train_epoch(
    model: TextToAudioEmbedding,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for text_labels, ground_truth in pbar:
        # Move to device
        ground_truth = ground_truth.to(device)  # (batch, 32, 1024)

        # Forward pass
        optimizer.zero_grad()
        prediction = model(text_labels)  # (batch, 32, 1024)

        # Compute loss (MSE between prediction and ground truth)
        loss = nn.functional.mse_loss(prediction, ground_truth)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: TextToAudioEmbedding,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for text_labels, ground_truth in tqdm(dataloader, desc="Validation"):
            # Move to device
            ground_truth = ground_truth.to(device)

            # Forward pass
            prediction = model(text_labels)

            # Compute loss
            loss = nn.functional.mse_loss(prediction, ground_truth)

            # Track loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train TextToAudioEmbedding (T5-based)")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing audio files and labels.json",
    )
    parser.add_argument(
        "--val-data-dir",
        type=str,
        default=None,
        help="Directory containing validation audio files and labels.json",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Determine device
    device = get_device(args.device if args.device != "auto" else "auto")

    logger.info("=" * 60)
    logger.info("🚀 Training TextToAudioEmbedding (T5-based)")
    logger.info("=" * 60)
    logger.info(f"   Device: {device}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.lr}")
    logger.info(f"   Seed: {args.seed}")
    logger.info(f"   Max duration: {args.max_duration}s")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize S3Tokenizer
    logger.info("📝 Loading S3Tokenizer...")
    s3_tokenizer = S3Tokenizer()
    s3_tokenizer = s3_tokenizer.to(device)
    s3_tokenizer.eval()
    logger.info(f"   ✓ Vocab size: {s3_tokenizer.vocab_size}")

    # Initialize SpeechTokenEmbedding
    logger.info("🎵 Loading SpeechTokenEmbedding...")
    speech_embedding = SpeechTokenEmbedding(
        vocab_size=s3_tokenizer.vocab_size,
        embedding_dim=1024,
    )
    speech_embedding = speech_embedding.to(device)
    speech_embedding.eval()

    # Initialize Perceiver Resampler
    logger.info("🗜️ Loading PerceiverResampler...")
    perceiver = PerceiverResampler(
        num_queries=32,
        query_dim=1024,
        embedding_dim=1024,
        num_heads=4,
    )
    perceiver = perceiver.to(device)
    perceiver.eval()
    logger.info("   ✓ Variable → 32 tokens")

    # Create dataset and dataloader
    logger.info(f"📂 Loading training data from: {args.data_dir}")
    train_dataset = AudioTextDataset(
        data_dir=Path(args.data_dir),
        s3_tokenizer=s3_tokenizer,
        speech_embedding=speech_embedding,
        perceiver=perceiver,
        device=device,
        max_duration=args.max_duration,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create validation dataloader if provided
    val_dataloader = None
    if args.val_data_dir:
        logger.info(f"📂 Loading validation data from: {args.val_data_dir}")
        val_dataset = AudioTextDataset(
            data_dir=Path(args.val_data_dir),
            s3_tokenizer=s3_tokenizer,
            speech_embedding=speech_embedding,
            perceiver=perceiver,
            device=device,
            max_duration=args.max_duration,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Initialize text encoder
    logger.info("📝 Initializing TextToAudioEmbedding...")
    text_encoder = TextToAudioEmbedding(device=device)

    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"📂 Loading checkpoint: {args.checkpoint}")
        text_encoder.load(args.checkpoint)

    # Initialize optimizer (only trainable params: projection + queries)
    trainable_params = text_encoder.get_trainable_params()
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    logger.info(f"   ✓ Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Training loop
    logger.info("=" * 60)
    logger.info("🏋️ Starting training...")
    logger.info("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(text_encoder, train_dataloader, optimizer, device)
        logger.info(f"   Train loss: {train_loss:.4f}")

        # Validate
        if val_dataloader is not None:
            val_loss = validate(text_encoder, val_dataloader, device)
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

    logger.info("=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"   Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
