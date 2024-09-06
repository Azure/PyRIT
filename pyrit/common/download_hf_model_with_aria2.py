# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
import logging
import subprocess
from huggingface_hub import HfApi

# Define the base directory for the project
PYRIT_PATH = pathlib.Path(__file__, "..", "..").resolve()

# Define the new folder for Hugging Face models within the same hierarchy as 'pyrit'
HF_MODELS_DIR = PYRIT_PATH / "hf_models"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

logger = logging.getLogger(__name__)

if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it before running this function.")

def get_available_files(model_id: str):
    """Fetches available files for a model from the Hugging Face repository."""
    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=HUGGINGFACE_TOKEN)
        available_files = [file.rfilename for file in model_info.siblings]
        return available_files
    except Exception as e:
        logger.info(f"Error fetching model files for {model_id}: {e}")
        return []

def download_files_with_aria2(urls: list, download_dir: str):
    """Uses aria2 to download files from the given list of URLs."""
    aria2_command = [
        "aria2c",
        "-d", download_dir,
        "-x", "3",
        "-s", "5",
        "-j", "4",
        "--continue=true",
        "--enable-http-pipelining=true",
        f"--header=Authorization: Bearer {HUGGINGFACE_TOKEN}",
        "-i", "-"  # Use '-' to read input from stdin
    ]

    try:
        # Run aria2c with input from stdin
        process = subprocess.Popen(aria2_command, stdin=subprocess.PIPE, text=True)
        process.communicate("\n".join(urls))  # Pass URLs directly to stdin
        if process.returncode == 0:
            logger.info(f"\nFiles downloaded successfully to {download_dir}.")
        else:
            logger.info(f"Error downloading files with aria2, return code: {process.returncode}.")
            raise subprocess.CalledProcessError(process.returncode, aria2_command)
    except subprocess.CalledProcessError as e:
        logger.info(f"Error downloading files with aria2: {e}")
        raise

def download_specific_files_with_aria2(model_id: str, file_patterns: list):
    """Downloads specific files from a Hugging Face model repository using aria2."""
    os.makedirs(HF_MODELS_DIR, exist_ok=True)

    available_files = get_available_files(model_id)
    files_to_download = [file for file in available_files if any(pattern in file for pattern in file_patterns)]

    if not files_to_download:
        logger.info(f"No files matched the patterns provided for model {model_id}.")
        return

    # Generate download URLs directly
    base_url = f"https://huggingface.co/{model_id}/resolve/main/"
    urls = [base_url + file for file in files_to_download]

    # Use the aria2c downloader without creating a .txt file
    download_files_with_aria2(urls, HF_MODELS_DIR)
