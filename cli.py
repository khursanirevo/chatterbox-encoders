"""
Command-line interface for chatterbox-encoders.

Usage:
    chatterbox-encode audio-to-tokens audio.wav
    chatterbox-encode audio-to-speaker audio.wav
    chatterbox-encode audio-prompt audio.wav --output prompt.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import librosa


def audio_to_tokens(args):
    """Convert audio to speech tokens."""
    from chatterbox_encoders import S3Tokenizer

    # Load tokenizer
    tokenizer = S3Tokenizer()

    # Load audio
    audio, sr = librosa.load(args.input, sr=16000)

    # Tokenize
    tokens, lengths = tokenizer.forward([audio], max_len=args.max_len)

    # Prepare output
    result = {
        "audio_path": str(args.input),
        "tokens": tokens[0].tolist(),
        "length": lengths[0].item(),
        "vocab_size": tokenizer.vocab_size,
        "token_rate": tokenizer.token_rate,
    }

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Tokens saved to: {args.output}")
    else:
        print(f"Tokens: {tokens.shape}")
        print(f"Length: {lengths[0].item()}")


def audio_to_speaker(args):
    """Extract speaker embedding from audio."""
    from chatterbox_encoders import VoiceEncoder

    # Load encoder
    ve = VoiceEncoder()

    if args.checkpoint:
        ve.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    ve.eval()

    # Load audio
    audio, sr = librosa.load(args.input, sr=16000)

    # Extract embedding
    embedding = ve.embeds_from_wavs([audio], sample_rate=16000, as_spk=True)

    # Prepare output
    result = {
        "audio_path": str(args.input),
        "speaker_embedding": embedding.tolist(),
        "embedding_dim": int(embedding.shape[0]),
    }

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Speaker embedding saved to: {args.output}")
    else:
        print(f"Speaker embedding: {embedding.shape}")


def audio_prompt(args):
    """Create complete audio prompt."""
    from chatterbox_encoders import (
        S3Tokenizer,
        VoiceEncoder,
        SpeakerProjector,
        EmotionProjector,
    )

    # Load models
    tokenizer = S3Tokenizer()
    ve = VoiceEncoder()

    if args.checkpoint:
        ve.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    ve.eval()

    speaker_proj = SpeakerProjector(256, 1024)
    emotion_proj = EmotionProjector()

    # Load audio
    audio, sr = librosa.load(args.input, sr=16000)
    duration = len(audio) / sr

    # Extract features
    speech_tokens, lengths = tokenizer.forward([audio])
    spk_emb = ve.embeds_from_wavs([audio], sample_rate=16000, as_spk=True)
    spk_emb_tensor = torch.from_numpy(spk_emb)
    spk_token = speaker_proj(spk_emb_tensor)
    emotion_token = emotion_proj(args.emotion)

    # Prepare output
    result = {
        "audio_path": str(args.input),
        "duration": duration,
        "speech_tokens": speech_tokens[0].tolist(),
        "speaker_embedding_256": spk_emb.tolist(),
        "speaker_token_1024": spk_token.squeeze(0).tolist(),
        "emotion_value": args.emotion,
        "emotion_token_1024": emotion_token.squeeze(0).tolist(),
    }

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Audio prompt saved to: {args.output}")
    else:
        print(f"Audio prompt created:")
        print(f"  - Duration: {duration:.2f}s")
        print(f"  - Speech tokens: {speech_tokens.shape}")
        print(f"  - Speaker embedding: {spk_emb.shape}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chatterbox Encoders - Extract features from audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # audio-to-tokens
    parser_tokens = subparsers.add_parser(
        "audio-to-tokens", help="Convert audio to speech tokens"
    )
    parser_tokens.add_argument("input", type=str, help="Input audio file")
    parser_tokens.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser_tokens.add_argument("--max-len", type=int, help="Maximum token length")

    # audio-to-speaker
    parser_speaker = subparsers.add_parser(
        "audio-to-speaker", help="Extract speaker embedding"
    )
    parser_speaker.add_argument("input", type=str, help="Input audio file")
    parser_speaker.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser_speaker.add_argument("--checkpoint", type=str, help="Voice encoder checkpoint")

    # audio-prompt
    parser_prompt = subparsers.add_parser(
        "audio-prompt", help="Create complete audio prompt"
    )
    parser_prompt.add_argument("input", type=str, help="Input audio file")
    parser_prompt.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser_prompt.add_argument("--checkpoint", type=str, help="Voice encoder checkpoint")
    parser_prompt.add_argument("--emotion", type=float, default=0.5, help="Emotion value (0.0-1.0)")

    # Parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "audio-to-tokens":
        audio_to_tokens(args)
    elif args.command == "audio-to-speaker":
        audio_to_speaker(args)
    elif args.command == "audio-prompt":
        audio_prompt(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
