# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import ChatMessage


class PromptTemplateGenerator:
    """A class to generate prompt templates based on specified types."""

    def __init__(self, type: str = "default"):
        self.type: str = type
        self.prompt_template: str = ""

    def generate_template(self, messages: list[ChatMessage]):
        """Generates a template based on the specified type. If no type is provided, it uses the default type.

        Raises:
            ValueError: Other than default template type is specified.

        Returns:
            str: The generated template content.
        """

        if self.type == "default":
            return self._generate_default_template(messages)
        else:
            raise ValueError(f"Unknown prompt template type: {type}")

    def _generate_default_template(self, messages: list[ChatMessage]):
        """Generates the default prompt template.

        Args:
            messages (list[ChatMessage]): list of chat messages
        """
        if not messages:
            raise ValueError("The messages list cannot be empty.")
        if not self.prompt_template:
            if len(messages) < 2:
                raise ValueError(
                    f"At least two chat message objects are required for the first call. Obtained only {len(messages)}."
                )
            for message in messages[:2]:
                self.prompt_template += message.role.upper() + ":" + message.content
        else:
            last_user_message = messages[-1]
            last_assistant_message = messages[-2]
            # update the prompt template with just assistant content since it has ASSISTANT marker
            # as part of the previous call
            self.prompt_template += last_assistant_message.content
            # update the assistant content since it has ASSISTANT marker part of the previous call
            self.prompt_template += last_user_message.role.upper() + ":" + last_user_message.content
        self.prompt_template += "ASSISTANT:"

        return self.prompt_template
