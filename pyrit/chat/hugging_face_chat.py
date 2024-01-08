# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from pyrit.common.prompt_template_generator import PromptTemplateGenerator
from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

logger = logging.getLogger(__name__)


class HuggingFaceChat(ChatSupport):
    """The HuggingFaceChat interacts with HuggingFace models, specifically for conducting red teaming activities.

    Args:
        ChatSupport (abc.ABC): Implementing methods for interactions with the HuggingFace model.
    """

    def __init__(
        self,
        *,
        model_id: str = "cognitivecomputations/WizardLM-7B-Uncensored",
        use_cuda: bool = False,
        tensor_format: str = "pt",
    ) -> None:
        """
        Args:
            model_id: HuggingFace model id which can be found in the model page. Defaults
            to cognitivecomputations/WizardLM-7B-Uncensored
            use_cuda: Flag to indicate whether to use CUDA (GPU) if available. It allows software
            developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing.
            tensor_format: Transformer models data tensor format, defaults to "pt" (PyTorch).
            "np" -> (Numpy) and "tf" ->TensorFlow
        """
        self.model_id: str = model_id
        self.use_cuda: bool = use_cuda

        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        # Load HuggingFace tokenizer and model
        self.tokenizer: AutoTokenizer = None
        self.model: AutoModelForCausalLM = None
        self.load_model_and_tokenizer()

        # Transformer models, data tensor formats, defaults to "pt" (PyTorch). "np" -> (Numpy) and "tf" ->TensorFlow
        self.tensor_format = tensor_format

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
            if self.use_cuda:
                self.model.to("cuda")  # Move the model to GPU
            logger.info(f"Model {self.model_id} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            raise

    def extract_last_assistant_response(self, text: str) -> str:
        """Identifies the last occurrence of 'ASSISTANT' in a given text string and extracts everything after it.

        Args::
            text (str): A string containing the conversation, including system instructions,
                    user inputs, and assistant responses, formatted with specific markers
                    and might contain a closing token '</s>'.

        Returns:
            str: The last response from the assistant in the provided text. If no assistant
            response is found, it returns an empty string.
        """

        # Find the last occurrence of "ASSISTANT:"
        last_assistant_index = text.rfind("ASSISTANT:")

        if last_assistant_index == -1:
            return ""

        # Extract the text after "ASSISTANT:"
        extracted_text = text[last_assistant_index + len("ASSISTANT:") :]

        # Find the closing token </s> and trim the text up to that point
        closing_token_index = extracted_text.find("</s>")
        if closing_token_index != -1:
            extracted_text = extracted_text[:closing_token_index].strip()

        return extracted_text

    async def complete_chat_async(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError

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
        prompt_template = self.prompt_template_generator.generate_template(messages)

        try:
            # Tokenize the chat template and obtain the input_ids
            input_ids = self.tokenizer(prompt_template, return_tensors=self.tensor_format).input_ids

            # Move the inputs to GPU for inferencing, if CUDA is available
            if self.use_cuda:
                input_ids = input_ids.to("cuda")  # Move the inputs to GPU for inferencing

            # Generate the response
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error occured during inference: {e}")
            raise

        # Clean the response message to remove specific tokens, if there are any
        extracted_response_message = self.extract_last_assistant_response(response_message)
        return extracted_response_message
