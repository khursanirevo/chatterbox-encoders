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
