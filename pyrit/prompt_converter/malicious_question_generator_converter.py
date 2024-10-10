# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import logging
import pathlib


from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.models import PromptRequestResponse, PromptRequestPiece, PromptTemplate
from pyrit.prompt_target import PromptTarget
from pyrit.common.path import DATASETS_PATH


# Use logger
logger = logging.getLogger(__name__)


class MaliciousQuestionGeneratorConverter(PromptConverter):
    """
    A PromptConverter that generates malicious questions using an LLM via an existing PromptTarget (like Azure OpenAI).
    """

    def __init__(self, target: PromptTarget, max_iterations: int = 10):
        """
        Initializes the MaliciousQuestionGeneratorConverter.

        Args:
            target (PromptTarget): The target to send prompts to (e.g., AzureOpenAICompletionTarget).
        """
        super().__init__()
        self.target = target

    def input_supported(self, input_type) -> bool:
        """
        Checks if the input type is supported by the converter.
        """
        return input_type == "text"

    async def convert_async(self, *, prompt: str, input_type="text") -> ConverterResult:
        """
        Converts the given prompt into malicious questions using the target LLM.
        Args:
            prompt (str): The prompt to be converted.
        Returns:
            ConverterResult: The result containing the generated malicious questions.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Load the YAML template
        prompt_template = PromptTemplate.from_yaml_file(
            pathlib.Path(DATASETS_PATH) / "prompt_converters" / "malicious_question_generator_converter.yaml"
        )

        # Apply the template with the given prompt
        formatted_prompt = prompt_template.apply_custom_metaprompt_parameters(prompt=prompt)

        # Build and send the prompt to generate one question
        prompt_request = await self._prepare_prompt_request(formatted_prompt)

        try:
            # Send prompt and handle response
            questions = await self._get_questions_from_response(prompt_request)

            # Return the first question if available
            if questions:
                return ConverterResult(output_text=questions[0], output_type="text")
            else:
                return ConverterResult(output_text="No question generated.", output_type="text")

        except Exception as e:
            logger.error(f"Error in MaliciousQuestionGeneratorConverter: {e}")
            return ConverterResult(output_text="Error generating questions", output_type="text")

    async def _prepare_prompt_request(self, formatted_prompt: str) -> PromptRequestResponse:
        """
        Prepares the prompt request to be sent to the LLM.
        """
        logger.info(f"Formatted Prompt: {formatted_prompt}")

        # Prepare request for sending to target using PromptRequestPiece
        prompt_request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value=formatted_prompt,
                    converted_value=formatted_prompt,
                )
            ]
        )
        logger.info(f"Prompt Request: {prompt_request}")
        return prompt_request

    async def _get_questions_from_response(self, prompt_request: PromptRequestResponse) -> list:
        """
        Sends the prompt to the LLM and parses the response into a list of questions.
        """
        # Send the prompt to the LLM via the target
        response_entry = await self.target.send_prompt_async(prompt_request=prompt_request)
        logger.info(f"Response Entry: {response_entry}")

        # Check if response contains 'response_text_pieces'
        if hasattr(response_entry, "response_text_pieces"):
            logger.info(f"Response Text Pieces: {response_entry.response_text_pieces}")

        # Parse the response and return the questions
        questions = self._parse_response(response_entry.request_pieces[0].converted_value)
        logger.info(f"Parsed Questions: {questions}")
        return questions

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
