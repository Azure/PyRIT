# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from pyrit.prompt_target.prompt_target import PromptTarget  
from pyrit.common.prompt_template_generator import PromptTemplateGenerator
from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_response import PromptRequestResponse, construct_response_from_request
from pyrit.models import ChatMessage


logger = logging.getLogger(__name__)


class HuggingFaceChatTarget(PromptTarget):
    """The HuggingFaceChatTarget interacts with HuggingFace models, specifically for conducting red teaming activities.
    Inherits from PromptTarget to comply with the current design standards.
    """

    def __init__(
        self,
        *,
        model_id: str = "cognitivecomputations/WizardLM-7B-Uncensored",
        use_cuda: bool = False,
        tensor_format: str = "pt",
        memory: MemoryInterface = None,
        verbose: bool = False
    ) -> None:
        super().__init__(memory=memory, verbose=verbose)
        self.model_id = model_id
        self.use_cuda = use_cuda
        self.tensor_format = tensor_format

        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        # Load the model and tokenizer using the encapsulated method
        self.load_model_and_tokenizer()

        # Initialize the PromptTemplateGenerator
        self.prompt_template_generator = PromptTemplateGenerator()


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
        """
        Load the model and tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            if self.use_cuda and torch.cuda.is_available():
                self.model.to("cuda")  # Move the model to GPU
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

        try:
            # Tokenize the chat template and obtain the input_ids
            input_ids = self.tokenizer(prompt_template, return_tensors=self.tensor_format).input_ids

            # Move the inputs to GPU for inferencing, if CUDA is available
            if self.use_cuda and torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            # Generate the response
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=400,
                temperature=1.0,
                top_p=1,
            )

            response_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_response = construct_response_from_request(
                request=request,
                response_text_pieces=[response_message],
                prompt_metadata={"model_id": self.model_id},
            )
            return prompt_response

        except Exception as e:
            logger.error(f"Error occurred during inference: {e}")
            raise


    def complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
    ) -> str:
        """Completes a chat interaction by generating a response to the given input prompt.
            Args:
                messages (list[ChatMessage]): The chat messages object containing the role and content.
                max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 400.
                temperature (float, optional): Controls randomness in the response generation. Defaults to 1.0.
                top_p (int, optional): Controls diversity of the response generation. Defaults to 1.
            Returns:
                str: The generated response message.
        """
        
        # Generate the prompt template using PromptTemplateGenerator
        prompt_template = self.prompt_template_generator.generate_template(messages)

        try:
            # Tokenize the chat template and obtain the input_ids
            input_ids = self.tokenizer(prompt_template, return_tensors=self.tensor_format).input_ids

            # Move the inputs to GPU for inferencing, if CUDA is available
            if self.use_cuda and torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            # Generate the response
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Error occurred during inference: {e}")
            raise

        # Clean the response message to remove specific tokens, if there are any
        extracted_response_message = self.extract_last_assistant_response(response_message)
        return extracted_response_message

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response.
        """
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")


    def extract_last_assistant_response(self, text: str) -> str:
        """
        Improved method to identify the last occurrence of 'ASSISTANT' in a given text string and extracts everything after it.
        """
        # Find the last occurrence of "ASSISTANT:"
        last_assistant_index = text.rfind("ASSISTANT:")

        if last_assistant_index == -1:
            return ""

        # Extract the text after "ASSISTANT:"
        extracted_text = text[last_assistant_index + len("ASSISTANT:"):]

        # Remove any extra spaces at the start of the extracted text
        extracted_text = extracted_text.lstrip()

        # Find the closing token </s> and trim the text up to that point
        closing_token_index = extracted_text.find("</s>")
        if closing_token_index != -1:
            extracted_text = extracted_text[:closing_token_index].strip()

        # Ensure no leading spaces are in the final extracted text
        final_text = extracted_text.strip()
        return final_text


    def _generate_prompt(self, messages: list[ChatMessage]) -> str:
        """
        Generates a prompt from a list of chat messages.
        """
        return "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)
