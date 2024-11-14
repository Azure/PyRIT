# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from pyrit.prompt_target import PromptChatTarget
from pyrit.common.download_hf_model import download_specific_files
from pyrit.models.prompt_request_response import PromptRequestResponse, construct_response_from_request
from pyrit.exceptions import EmptyResponseException, pyrit_target_retry
from pyrit.common import default_values


logger = logging.getLogger(__name__)


class HuggingFaceChatTarget(PromptChatTarget):
    """The HuggingFaceChatTarget interacts with HuggingFace models, specifically for conducting red teaming activities.
    Inherits from PromptTarget to comply with the current design standards.
    """

    # Class-level cache for model and tokenizer
    _cached_model = None
    _cached_tokenizer = None
    _cached_model_id = None

    # Class-level flag to enable or disable cache
    _cache_enabled = True

    # Define the environment variable name for the Hugging Face token
    HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE = "HUGGINGFACE_TOKEN"

    def __init__(
        self,
        *,
        model_id: str,
        hf_access_token: Optional[str] = None,
        use_cuda: bool = False,
        tensor_format: str = "pt",
        necessary_files: list = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        skip_special_tokens: bool = True,
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.use_cuda = use_cuda
        self.tensor_format = tensor_format

        # Use the `get_required_value` to get the API key (from env or passed value)
        self.huggingface_token = default_values.get_required_value(
            env_var_name=self.HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE, passed_value=hf_access_token
        )

        try:
            import torch
        except ModuleNotFoundError as e:
            logger.error("Could not import torch. You may need to install it via 'pip install pyrit[all]'")
            raise e

        # Determine the device
        self.device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Set necessary files if provided, otherwise set to None to trigger general download
        self.necessary_files = necessary_files

        # Set the default parameters for the model generation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.skip_special_tokens = skip_special_tokens

        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        self.load_model_and_tokenizer_task = asyncio.create_task(self.load_model_and_tokenizer())

    def is_model_id_valid(self) -> bool:
        """
        Check if the HuggingFace model ID is valid.
        :return: True if valid, False otherwise.
        """
        try:
            # Attempt to load the configuration of the model
            PretrainedConfig.from_pretrained(self.model_id)
            return True
        except Exception as e:
            logger.error(f"Invalid HuggingFace model ID {self.model_id}: {e}")
            return False

    async def load_model_and_tokenizer(self):
        """Loads the model and tokenizer, downloading if necessary.

        Downloads the model to the HF_MODELS_DIR folder if it does not exist,
        then loads it from there.

        Raises:
            Exception: If the model loading fails.
        """
        try:
            # Define the default Hugging Face cache directory
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{self.model_id.replace('/', '--')}"
            )

            # Check if the model is already cached
            if HuggingFaceChatTarget._cache_enabled and HuggingFaceChatTarget._cached_model_id == self.model_id:
                logger.info(f"Using cached model and tokenizer for {self.model_id}.")
                self.model = HuggingFaceChatTarget._cached_model
                self.tokenizer = HuggingFaceChatTarget._cached_tokenizer
                return

            if self.necessary_files is None:
                # Download all files if no specific files are provided
                logger.info(f"Downloading all files for {self.model_id}...")
                await download_specific_files(self.model_id, None, self.huggingface_token, cache_dir)
            else:
                # Download only the necessary files
                logger.info(f"Downloading specific files for {self.model_id}...")
                await download_specific_files(self.model_id, self.necessary_files, self.huggingface_token, cache_dir)

            # Load the tokenizer and model from the specified directory
            logger.info(f"Loading model {self.model_id} from cache path: {cache_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir=cache_dir)

            # Move the model to the correct device
            self.model = self.model.to(self.device)

            # Debug prints to check types
            logger.info(f"Model loaded: {type(self.model)}")  # Debug print
            logger.info(f"Tokenizer loaded: {type(self.tokenizer)}")  # Debug print

            # Cache the loaded model and tokenizer if caching is enabled
            if HuggingFaceChatTarget._cache_enabled:
                HuggingFaceChatTarget._cached_model = self.model
                HuggingFaceChatTarget._cached_tokenizer = self.tokenizer
                HuggingFaceChatTarget._cached_model_id = self.model_id

            logger.info(f"Model {self.model_id} loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            raise

    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt asynchronously to the HuggingFace model.
        """
        # Load the model and tokenizer using the encapsulated method
        await self.load_model_and_tokenizer_task

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt_template = request.converted_value

        logger.info(f"Sending the following prompt to the HuggingFace model: {prompt_template}")

        # Prepare the input messages using chat templates
        messages = [{"role": "user", "content": prompt_template}]

        # Apply chat template via the _apply_chat_template method
        tokenized_chat = self._apply_chat_template(messages)
        input_ids = tokenized_chat["input_ids"]
        attention_mask = tokenized_chat["attention_mask"]

        logger.info(f"Tokenized chat: {input_ids}")

        try:
            # Ensure model is on the correct device (should already be the case from `load_model_and_tokenizer`)
            self.model.to(self.device)

            # Record the length of the input tokens to later extract only the generated tokens
            input_length = input_ids.shape[-1]

            # Generate the response
            logger.info("Generating response from model...")
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            logger.info(f"Generated IDs: {generated_ids}")  # Log the generated IDs

            # Extract the assistant's response by slicing the generated tokens after the input tokens
            generated_tokens = generated_ids[0][input_length:]

            # Decode the assistant's response from the generated token IDs
            assistant_response = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=self.skip_special_tokens
            ).strip()

            if not assistant_response:
                raise EmptyResponseException()

            logger.info(f"Assistant's response: {assistant_response}")

            return construct_response_from_request(
                request=request,
                response_text_pieces=[assistant_response],
                prompt_metadata=json.dumps({"model_id": self.model_id}),
            )

        except Exception as e:
            logger.error(f"Error occurred during inference: {e}")
            raise

    def _apply_chat_template(self, messages):
        """
        A private method to apply the chat template to the input messages and tokenize them.
        """
        # Check if the tokenizer has a chat template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            logger.info("Tokenizer has a chat template. Applying it to the input messages.")

            # Apply the chat template to format and tokenize the messages
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=self.tensor_format,
                return_dict=True,
            ).to(self.device)
            return tokenized_chat
        else:
            error_message = (
                "Tokenizer does not have a chat template. "
                "This model is not supported, as we only support instruct models with a chat template."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response.
        """
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    @classmethod
    def enable_cache(cls):
        """Enables the class-level cache."""
        cls._cache_enabled = True
        logger.info("Class-level cache enabled.")

    @classmethod
    def disable_cache(cls):
        """Disables the class-level cache and clears the cache."""
        cls._cache_enabled = False
        cls._cached_model = None
        cls._cached_tokenizer = None
        cls._cached_model_id = None
        logger.info("Class-level cache disabled and cleared.")
