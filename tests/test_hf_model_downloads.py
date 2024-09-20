# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import pytest

# Import functions to test from local application files
from pyrit.common.download_hf_model_with_aria2 import download_specific_files_with_aria2
from pyrit.common.download_hf_model_with_hf_cli import (
    download_specific_files_with_cli,
)
from pyrit.common.download_hf_model_with_transformers import download_model_with_transformers


# Define constants for testing
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
FILE_PATTERNS = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
]


@pytest.fixture(scope="module")
def setup_environment():
    """Fixture to set up the environment for Hugging Face downloads."""
    # Check for Hugging Face token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    yield  # Yield control back to the tests after setup


def test_download_specific_files_with_aria2(setup_environment):
    """Test downloading specific files using aria2."""
    start_time = time.time()
    try:
        # The function is responsible for folder creation.
        download_specific_files_with_aria2(MODEL_ID, FILE_PATTERNS)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for aria2 specific file download: {elapsed_time:.2f} seconds")
    except Exception as e:
        pytest.fail(f"aria2 specific file download failed: {e}")


def test_download_specific_files_with_cli(setup_environment):
    """Test downloading specific files using huggingface-cli and measure time."""
    start_time = time.time()
    try:
        # The function is responsible for folder creation.
        download_specific_files_with_cli(MODEL_ID, FILE_PATTERNS)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for huggingface-cli specific file download: {elapsed_time:.2f} seconds")
    except Exception as e:
        pytest.fail(f"Specific file download failed: {e}")


def test_download_model_with_transformers(setup_environment):
    """Test downloading model using transformers library and measure time."""
    start_time = time.time()
    try:
        # The function is responsible for folder creation.
        tokenizer, model = download_model_with_transformers(MODEL_ID)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for transformers model download: {elapsed_time:.2f} seconds")
        assert tokenizer is not None and model is not None, "Tokenizer and model should be loaded."
    except Exception as e:
        pytest.fail(f"Model download failed: {e}")
