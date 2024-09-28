# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest
from unittest.mock import patch

# Import functions to test from local application files
from pyrit.common.download_hf_model_with_aria2 import download_specific_files_with_aria2


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
    with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "mocked_token"}):
        token = os.getenv("HUGGINGFACE_TOKEN")
        yield token


def test_download_specific_files_with_aria2(setup_environment):
    """Test downloading specific files using aria2."""
    token = setup_environment  # Get the token from the fixture
    with pytest.raises(Exception):
        download_specific_files_with_aria2(MODEL_ID, FILE_PATTERNS, token)
