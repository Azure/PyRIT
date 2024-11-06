# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import urllib.request
from pathlib import Path

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


def download_specific_files(model_id: str, file_patterns: list, token: str, cache_dir: Path) -> list[str]:
    """
    Downloads specific files from a Hugging Face model repository.
    If file_patterns is None, downloads all files.

    Returns:
        List of URLs for the downloaded files.
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

    # Download the files
    download_files(urls, token, cache_dir)


def download_files(urls: list, token: str, cache_dir: Path):
    headers = {"Authorization": f"Bearer {token}"}
    for url in urls:
        local_filename = Path(cache_dir, url.split("/")[-1])
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response, open(local_filename, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        logger.info(f"Downloaded {local_filename}")
