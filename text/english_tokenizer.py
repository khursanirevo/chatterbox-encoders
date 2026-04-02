"""
English text tokenizer for Chatterbox.

Tokenizes English text using the 🤗 Tokenizers library.
"""

import json
import logging
from pathlib import Path
from typing import Union

import torch
from tokenizers import Tokenizer

from chatterbox_encoders.config.constants import (
    SOT,
    EOT,
    UNK,
    SPACE,
)

logger = logging.getLogger(__name__)


class EnTokenizer:
    """
    English text tokenizer.

    Uses 🤗 Tokenizers library with a custom vocabulary optimized for
    Chatterbox TTS.

    Args:
        vocab_file_path: Path to tokenizer.json file

    Examples:
        >>> tokenizer = EnTokenizer("tokenizer.json")
        >>> tokens = tokenizer.text_to_tokens("Hello world")
        >>> tokens.shape
        torch.Size([1, num_tokens])
    """

    def __init__(self, vocab_file_path: str):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        """Verify special tokens exist in vocabulary."""
        voc = self.tokenizer.get_vocab()
        assert SOT in voc, f"Start token {SOT} not in vocabulary"
        assert EOT in voc, f"End token {EOT} not in vocabulary"

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert text to tokens.

        Args:
            text: Input text string

        Returns:
            torch.Tensor: Token IDs (1, seq_len)

        Examples:
            >>> tokenizer = EnTokenizer("tokenizer.json")
            >>> tokens = tokenizer.text_to_tokens("Hello world")
        """
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str) -> list:
        """
        Encode text to token IDs.

        Args:
            txt: Text string

        Returns:
            list: List of token IDs

        Examples:
            >>> tokenizer = EnTokenizer("tokenizer.json")
            >>> ids = tokenizer.encode("Hello world")
        """
        txt = txt.replace(" ", SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq: torch.Tensor) -> str:
        """
        Decode token IDs back to text.

        Args:
            seq: Token IDs tensor (1, T) or (T,)

        Returns:
            str: Decoded text

        Examples:
            >>> tokenizer = EnTokenizer("tokenizer.json")
            >>> tokens = torch.tensor([[1, 2, 3]])
            >>> text = tokenizer.decode(tokens)
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(" ", "")
        txt = txt.replace(SPACE, " ")
        txt = txt.replace(EOT, "")
        txt = txt.replace(UNK, "")
        return txt
