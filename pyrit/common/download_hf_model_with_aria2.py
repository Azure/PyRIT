# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from pathlib import Path
import subprocess
from typing import Optional

from huggingface_hub import HfApi


logger = logging.getLogger(__name__)


def get_available_files(model_id: str, token: str):
    """Fetches available files for a model from the Hugging Face repository."""
    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token)
        available_files = [file.rfilename for file in model_info.siblings]

        # Perform simple validation: raise a ValueError if no files are available
        if not len(available_files):
            raise ValueError(f"No available files found for the model: {model_id}")

        return available_files
    except Exception as e:
        logger.info(f"Error fetching model files for {model_id}: {e}")
        return []


def download_files_with_aria2(urls: list, token: str, download_dir: Optional[Path] = None):
    """Uses aria2 to download files from the given list of URLs."""

    # Convert download_dir to string if it's a Path object
    download_dir_str = str(download_dir) if isinstance(download_dir, Path) else download_dir

    aria2_command = [
        "aria2c",
        "-d",
        download_dir_str,
        "-x",
        "3",  # Number of connections per server for each download.
        "-s",
        "5",  # Number of splits for each file.
        "-j",
        "4",  # Maximum number of parallel downloads.
        "--continue=true",
        "--enable-http-pipelining=true",
        f"--header=Authorization: Bearer {token}",
        "-i",
        "-",  # Use '-' to read input from stdin
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


def download_specific_files_with_aria2(model_id: str, file_patterns: list, token: str, cache_dir: Optional[Path]):
    """
    Downloads specific files from a Hugging Face model repository using aria2.
    If file_patterns is None, downloads all files.
    """
    os.makedirs(cache_dir, exist_ok=True)

    available_files = get_available_files(model_id, token)
    # If no file patterns are provided, download all available files
    if file_patterns is None:
        files_to_download = available_files
        logger.info(f"Downloading all files for model {model_id}.")
    else:
        # Filter files based on the patterns provided
        files_to_download = [file for file in available_files if any(pattern in file for pattern in file_patterns)]
        if not files_to_download:
            logger.info(f"No files matched the patterns provided for model {model_id}.")
            return

    # Generate download URLs directly
    base_url = f"https://huggingface.co/{model_id}/resolve/main/"
    urls = [base_url + file for file in files_to_download]

    # Use aria2c to download the files
    download_files_with_aria2(urls, token, cache_dir)
