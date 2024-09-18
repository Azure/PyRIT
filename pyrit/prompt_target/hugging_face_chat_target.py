# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig


from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.common.dynamic_prompt_formatter import format_prompt
from pyrit.common.download_hf_model_with_hf_cli import download_model_with_cli, download_specific_files_with_cli
from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_response import PromptRequestResponse, construct_response_from_request


logger = logging.getLogger(__name__)


class HuggingFaceChatTarget(PromptTarget):
    """The HuggingFaceChatTarget interacts with HuggingFace models, specifically for conducting red teaming activities.
    Inherits from PromptTarget to comply with the current design standards.
    """

    # Class-level cache for model and tokenizer
    _cached_model = None
    _cached_tokenizer = None
    _cached_model_id = None

    # Class-level flag to enable or disable cache
    _cache_enabled = False

    def __init__(
        self,
        *,
        model_id: str,
        use_cuda: bool = False,
        tensor_format: str = "pt",
        memory: MemoryInterface = None,
        verbose: bool = False,
        necessary_files: list = None,
        max_new_tokens: int = 20,      # Default max_new_tokens parameter
        temperature: float = 1.0,      # Default temperature parameter
        top_p: float = 1.0,  
    ) -> None:
        super().__init__(memory=memory, verbose=verbose)
        self.model_id = model_id
        self.use_cuda = use_cuda
        self.tensor_format = tensor_format

        # Determine the device
        self.device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Set necessary files if provided, otherwise set to None to trigger general download
        self.necessary_files = necessary_files

        # Set the default parameters for the model generation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        # Load the model and tokenizer using the encapsulated method
        self.load_model_and_tokenizer()

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

    def load_model_and_tokenizer(self):
        """Loads the model and tokenizer, downloading if necessary.

            Downloads the model to the HF_MODELS_DIR folder if it does not exist,
            then loads it from there.

            Raises:
                Exception: If the model loading fails.
        """
        try:
           # Check if the model is already cached
            if HuggingFaceChatTarget._cache_enabled and HuggingFaceChatTarget._cached_model_id == self.model_id:
                logger.info(f"Using cached model and tokenizer for {self.model_id}.")
                self.model = HuggingFaceChatTarget._cached_model
                self.tokenizer = HuggingFaceChatTarget._cached_tokenizer
                return

            # Define the default Hugging Face cache directory
            cache_dir = os.path.join(os.path.expanduser("~"),".cache","huggingface","hub",f"models--{self.model_id.replace('/', '--')}")

            if self.necessary_files is None:
                # Perform general download if no specific files are mentioned
                logger.info(f"Downloading the entire model {self.model_id} since no specific files are provided...")
                download_model_with_cli(self.model_id)
            else:
                # Check if the necessary files are already in the Hugging Face cache
                missing_files = [file for file in self.necessary_files if not os.path.exists(os.path.join(cache_dir, file))]

                if missing_files:
                    # If some files are missing, use CLI to download them
                    logger.info(f"Model {self.model_id} not fully found in cache. Downloading missing files using CLI...")
                    download_specific_files_with_cli(self.model_id, missing_files)  # Download only the missing files

            # Load the tokenizer and model from the specified directory
            logger.info(f"Loading model {self.model_id} from cache path: {cache_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
            #self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", cache_dir=cache_dir)
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

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt asynchronously to the HuggingFace model.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt_template = request.converted_value

        logger.info(f"Sending the following prompt to the HuggingFace model: {prompt_template}")

        # Prepare the input messages using chat templates
        messages = [{"role": "user", "content": prompt_template}]

        # Check if the tokenizer has a chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            logger.info("Tokenizer has a chat template. Applying it to the input messages.")

            # Apply the chat template to format and tokenize the messages
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=self.tensor_format
            ).to(self.device)
        else:
            logger.info("Tokenizer does not have a chat template. Using default formatting.")

            # Format the prompt dynamically based on the tokenizer configuration
            prompt_text = format_prompt(self.tokenizer, prompt_template)

            # Tokenize the prompt
            tokenized_chat = self.tokenizer(
                prompt_text,
                return_tensors=self.tensor_format,
                add_special_tokens=False  # We manage special tokens manually
            ).input_ids.to(self.device)

        logger.info(f"Tokenized chat: {tokenized_chat}")

        try:
            # Ensure model is on the correct device (should already be the case from `load_model_and_tokenizer`)
            self.model.to(self.device)

            # Record the length of the input tokens to later extract only the generated tokens
            input_length = tokenized_chat.shape[-1]

            # Generate the response
            logger.info("Generating response from model...")
            generated_ids = self.model.generate(
                input_ids=tokenized_chat,  
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            logger.info(f"Generated IDs: {generated_ids}")  # Log the generated IDs

             # Extract the assistant's response by slicing the generated tokens after the input tokens
            generated_tokens = generated_ids[0][input_length:]

            # Decode the assistant's response from the generated token IDs
            assistant_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            logger.info(f"Assistant's response: {assistant_response}")

            prompt_response = construct_response_from_request(
                request=request,
                response_text_pieces=[assistant_response],
                prompt_metadata=json.dumps({"model_id": self.model_id}),
            )

        except Exception as e:
            logger.error(f"Error occurred during inference: {e}")
            raise

        return prompt_response
    
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