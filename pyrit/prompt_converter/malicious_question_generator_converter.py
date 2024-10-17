# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import logging
import pathlib


from pyrit.prompt_converter import LLMGenericTextConverter, ConverterResult
from pyrit.models import PromptRequestResponse, PromptRequestPiece, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import DATASETS_PATH


# Use logger
logger = logging.getLogger(__name__)


class MaliciousQuestionGeneratorConverter(LLMGenericTextConverter):
    """
    A PromptConverter that generates malicious questions using an LLM via an existing PromptTarget (like Azure OpenAI).
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: PromptTemplate, **kwargs):
        """
        Initializes the converter with a specific target and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (PromptTemplate): The prompt template to use.
        """
        # Call the parent constructor to initialize the LLMGenericTextConverter
        super().__init__(converter_target=converter_target, prompt_template=prompt_template, **kwargs)

    async def convert_async(self, *, prompt: str, input_type="text") -> ConverterResult:
        """
        Overrides the convert_async method to include custom cleaning and parsing
        of the LLM's response after the prompt has been processed.

        Args:
            prompt (str): The prompt to convert.
            input_type (PromptDataType): The input data type (default is "text").
        """

        # Call the parent class's convert_async to handle the prompt conversion
        result = await super().convert_async(prompt=prompt, input_type=input_type)

        # Clean and parse the result output_text (the LLM's response)
        cleaned_response = self._clean_response(result.output_text)
        parsed_questions = self._parse_response(cleaned_response)

        # Return the parsed result as a ConverterResult object
        return ConverterResult(
            output_text=parsed_questions[0] if parsed_questions else "No question generated.",
            output_type="text"
        )

    def _clean_response(self, response: str) -> str:
        """Cleans the LLM response by removing code block markers and extraneous text."""
        # Remove code block markers and clean response
        cleaned_response = response.replace("```python", "").replace("```", "").strip()

        # If the response starts with 'questions =', remove that part
        if cleaned_response.startswith("questions ="):
            cleaned_response = cleaned_response[len("questions =") :].strip()

        return cleaned_response

    def _parse_response(self, response: str) -> list[str]:
        """Parses the cleaned LLM response into a Python list of questions."""
        try:
            # Clean the response
            cleaned_response = self._clean_response(response)

            # Use ast.literal_eval to safely evaluate the string as a Python literal
            parsed_list = ast.literal_eval(cleaned_response)

            # Ensure the result is a list
            if isinstance(parsed_list, list):
                # Clean up the individual questions by stripping whitespace
                questions = [q.strip() for q in parsed_list]
                return questions
            else:
                raise ValueError("The response is not a valid Python list.")

        except (SyntaxError, ValueError) as e:
            logger.error(f"Failed to parse LLM response as Python list: {e}")
            return ["Error parsing response."]
