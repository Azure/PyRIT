# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_model_with_transformers(model_id: str):
    """Downloads a model using transformers library with a progress bar.
    
    Args:
        model_id: The model ID from Hugging Face.
        cache_dir: The cache directory where the model should be downloaded.
        
    Returns:
        Tuple of the loaded tokenizer and model.
    """
    try:
        # Set up the progress bar
        hf_logging.set_verbosity_info()  # Ensure the logging level allows progress bar output
        hf_logging.enable_progress_bar()
        
        # Load the tokenizer and model with progress bar
        logger.info(f"Downloading model and tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        logger.info(f"Model and tokenizer for {model_id} downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading model with transformers: {e}")
        raise

    return tokenizer, model
