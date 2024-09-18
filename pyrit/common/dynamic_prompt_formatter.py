# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

# Set up logging configuration
logger = logging.getLogger(__name__)


def format_prompt(tokenizer, prompt_template):
    """
    Formats the prompt based on the tokenizer's configuration.

    Args:
        tokenizer: The tokenizer instance.
        prompt_template: The user's prompt as a string.

    Returns:
        A formatted prompt string.
    """
    try:
        # Get tokenizer class
        tokenizer_class = tokenizer.__class__.__name__
        logger.info(f"Formatting prompt for tokenizer class: {tokenizer_class}")

        # Get special tokens
        special_tokens = tokenizer.special_tokens_map
        logger.debug(f"Special tokens available: {special_tokens}")

        # Initialize prompt_text with the original template
        prompt_text = prompt_template

        # Manually format the prompt based on tokenizer class and special tokens
        if tokenizer_class in ['LlamaTokenizer', 'LlamaTokenizerFast']:
            logger.info(f"Applying LLaMA-specific formatting.")
            # For LLaMA-based models
            bos_token = special_tokens.get('bos_token', '<s>')
            eos_token = special_tokens.get('eos_token', '</s>')
            add_eos = special_tokens.get('add_eos_token', False)
            logger.debug(f"Using bos_token: {bos_token}, eos_token: {eos_token}, add_eos: {add_eos}")

            prompt_text = f"{bos_token} {prompt_template}{' ' + eos_token if add_eos else ''}"

        elif tokenizer_class == 'GemmaTokenizer':
            logger.info(f"Applying Gemma-specific formatting.")
            # For GemmaTokenizer-based models
            start_turn = "<start_of_turn>"
            end_turn = "<end_of_turn>"
            bos_token = special_tokens.get('bos_token', '<bos>')
            eos_token = special_tokens.get('eos_token', '<eos>')
            logger.debug(f"Using start_turn: {start_turn}, bos_token: {bos_token}, eos_token: {eos_token}, end_turn: {end_turn}")

            # Add turn markers and end markers
            prompt_text = f"{start_turn} {bos_token} {prompt_template} {eos_token} {end_turn}"

        else:
            logger.warning(f"Unknown tokenizer class '{tokenizer_class}'. Applying default formatting.")
            # Default formatting if no specific handling is defined
            bos_token = special_tokens.get('bos_token', '')
            eos_token = special_tokens.get('eos_token', '')
            prompt_text = f"{bos_token} {prompt_template} {eos_token}"

        logger.info(f"Formatted prompt: {prompt_text}")
        return prompt_text.strip()

    except Exception as e:
        logger.error(f"Error during prompt formatting: {e}")
        raise

