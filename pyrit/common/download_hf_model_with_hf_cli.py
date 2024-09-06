# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import subprocess
from pyrit.common import default_values
from huggingface_hub import HfApi  # Hugging Face API for model size

logger = logging.getLogger(__name__)

# Load environment variables
default_values.load_default_env()


# Set environment variables for huggingface-cli
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable parallel downloads


def login_to_huggingface():
    """Logs into Hugging Face using the token provided as an environment variable."""
    token = os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it before running this function.")

    command = [
        "huggingface-cli", 
        "login", 
        "--token", 
        token, 
        "--add-to-git-credential"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Successfully logged into Hugging Face.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error logging into Hugging Face: {e}")
        raise


def get_available_files(model_id: str):
    """Fetches available files for a model from the Hugging Face repository."""
    api = HfApi()
    try:
        model_info = api.model_info(model_id)
        available_files = [file.rfilename for file in model_info.siblings]
        return available_files
    except Exception as e:
        logger.info(f"Error fetching model files for {model_id}: {e}")
        return []


def download_model_with_cli(model_id: str):
    """Downloads a Hugging Face model using huggingface-cli with parallel transfer enabled.

        Args:
            model_id: The model ID from Hugging Face.
        Raises:
            subprocess.CalledProcessError: If the huggingface-cli command fails.
    """
    logger.info("Please make sure you are logged in to Hugging Face CLI. Running login process...")
    login_to_huggingface()  # Log in before downloading

    # Command to run huggingface-cli
    command = [
        "huggingface-cli",
        "download",
        model_id,
        "--repo-type", "model"  # Ensure we specify the repository type
    ]

    try:
        # Run the huggingface-cli command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Model {model_id} downloaded successfully to the cache.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error downloading model {model_id}: {e}")
        raise


def download_specific_files_with_cli(model_id: str, file_patterns: list):
    """Downloads specific files from a Hugging Face model repository using huggingface-cli.

        Args:
            model_id: The model ID from Hugging Face.
            file_patterns: A list of file patterns to download.
        Raises:
            subprocess.CalledProcessError: If the huggingface-cli command fails.
    """
    logger.info("Please make sure you are logged in to Hugging Face CLI. Running login process...")
    login_to_huggingface()

    available_files = get_available_files(model_id)
    files_to_download = [file for file in available_files if any(pattern in file for pattern in file_patterns)]

    if not files_to_download:
        logger.info(f"No files matched the patterns provided for model {model_id}.")
        return

    logger.info("The following files will be downloaded:")
    for file in files_to_download:
        logger.info(f"  - {file}")

    command = [
        "huggingface-cli",
        "download",
        model_id,
        "--repo-type", "model",
        "--include", ", ".join(files_to_download)
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        downloaded_files = [line.split()[-1] for line in result.stdout.splitlines() if "Downloading" in line]

        logger.info("Downloaded files:")
        for file in downloaded_files:
            logger.info(f"  - {file}")
        logger.info(f"Specified files for {model_id} downloaded successfully to the cache.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error downloading specific files for model {model_id}: {e}")
        logger.info(e.stderr)
        raise
