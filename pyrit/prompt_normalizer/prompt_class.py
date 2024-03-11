# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter


class Prompt(abc.ABC):
    memory: MemoryInterface

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: list[PromptConverter],
        prompt_text: str,
        conversation_id: str,
    ) -> None:
        """
        Initialize a PromptClass object.

        Args:
            prompt_target (PromptTarget): The target for the prompt.
            prompt_converters (list[PromptConverter]): A list of prompt converters.
            prompt_text (str): The text of the prompt.
            conversation_id (str): The ID of the conversation.

        Raises:
            ValueError: If any of the arguments are of incorrect type.

        """
        if not isinstance(prompt_target, PromptTarget):
            raise ValueError("prompt_target must be a PromptTarget")

        if (
            not isinstance(prompt_converters, list)
            or len(prompt_converters) == 0
            or not all(isinstance(converter, PromptConverter) for converter in prompt_converters)
        ):
            raise ValueError("prompt_converters must be a list[PromptConverter] and be non-empty")

        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be a str")

        if not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a str")

        self._prompt_target = prompt_target
        self._prompt_converters = prompt_converters
        self._prompt_text = prompt_text
        self.conversation_id = conversation_id

    def send_prompt(self, normalizer_id: str) -> list[str]:
        """
        Sends the prompt to the prompt target, by first converting the prompt.
        The prompt runs through every converter (the output of one converter is
        the input of the next converter).
        """
        converted_prompts = [self._prompt_text]

        for converter in self._prompt_converters:
            converted_prompts = converter.convert(converted_prompts)

        responses = []
        for converted_prompt in converted_prompts:
            responses.append(
                self._prompt_target.send_prompt(
                    normalized_prompt=converted_prompt,
                    conversation_id=self.conversation_id,
                    normalizer_id=normalizer_id,
                )
            )
        return responses
