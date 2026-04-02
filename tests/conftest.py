"""
Pytest configuration for chatterbox_encoders tests.
"""

import os
import sys

# Force CPU-only mode before any torch imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def pytest_configure():
    """Configure pytest to handle torch CUDA issues."""
    # Set environment variables before any imports
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''
