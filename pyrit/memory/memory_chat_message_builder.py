# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from itertools import groupby

from pyrit.memory.memory_models import PromptMemoryEntry
from pyrit.models import ChatMessage
from pyrit.memory.memory_models import PromptMemoryEntry

class MemoryChatMessageBuilder:
    """
    A class for composing chat messages based on memory entries.
    """
    def __init__(self):
        self._has_multimodal_input = None

    def build_chat_messages(self, memory_entries: list[PromptMemoryEntry]) -> list[ChatMessage]:
        """
        Builds chat messages based on memory entries.

        Args:
            memory_entries (list[PromptMemoryEntry]): A list of PromptMemoryEntry objects.

        Returns:
            list[ChatMessage]: The list of constructed chat messages.
        """
        if self._check_multimodal_input(memory_entries):
            target_names = set(me.prompt_target_identifier["__type__"] for me in memory_entries)
            if len(target_names) == 1 and "AzureOpenAIChatTarget" in target_names:
                return self._construct_multimodal_messages(memory_entries)
            else:
                raise ValueError("Multimodal inputs are not yet supported for this target: " + ", ".join(target_names))
        else:
            return self._construct_text_messages(memory_entries)

    def _check_multimodal_input(self, memory_entries: list[PromptMemoryEntry]) -> bool:
        """
        Checks if the given memory entries contain multimodal input.

        Args:
            memory_entries (list[PromptMemoryEntry]): A list of PromptMemoryEntry objects.

        Returns:
            bool: True if the memory entries contain multimodal input, False otherwise.
        """
        if len(memory_entries) == 1 and memory_entries[0].role == "system":
            return False
        
        if self._has_multimodal_input is not None:
            return self._has_multimodal_input
        
        # Dictionary to store the original_prompt_data_type for each sequence
        user_data_types = {}
        for prompt_req_resp_obj in memory_entries:
            if prompt_req_resp_obj.role == "user":
                key = prompt_req_resp_obj.sequence
                if key in user_data_types:
                    if user_data_types[key] != prompt_req_resp_obj.original_prompt_data_type:
                        self._has_multimodal_input = True
                        return True
                else:
                    user_data_types[key] = prompt_req_resp_obj.original_prompt_data_type
        self._has_multimodal_input = False
        return False

    def _construct_multimodal_messages(self, memory_entries: list[PromptMemoryEntry]) -> list[ChatMessage]:
        """
        Constructs chat messages from the given memory entries.

        Args:
            memory_entries (list[PromptMemoryEntry]): A list of PromptMemoryEntry objects.

        Returns:
            list[ChatMessage]: The list of constructed chat messages.
        """
        # Grouping objects by sequence
        grouped_objects = {}
        for key, group in groupby(sorted(memory_entries, key=lambda x: x.sequence), key=lambda x: x.sequence):
            grouped_objects[key] = list(group)

        # Combining objects in the same sequence and storing in a list
        chat_messages = []
        for key, group in sorted(grouped_objects.items()):
            content = []
            role = None
            for prompt_memory_entry in group:
                if not role:
                    role = prompt_memory_entry.role
                if prompt_memory_entry.original_prompt_data_type == "text":
                    entry = {"type": prompt_memory_entry.original_prompt_data_type, 
                            "text": prompt_memory_entry.original_prompt_text}
                    content.append(entry)
                elif prompt_memory_entry.original_prompt_data_type == "image_url":
                    entry = {"type": prompt_memory_entry.original_prompt_data_type,
                         "image_url": prompt_memory_entry.original_prompt_text}
                    content.append(entry)
                else:
                    raise ValueError(f"Multimodal data type {prompt_memory_entry.original_prompt_data_type} is not yet supported.")
            chat_message = ChatMessage(role=role, content=content)
            chat_messages.append(chat_message)
        return chat_messages

    def _construct_text_messages(self, memory_entries):
        return [ChatMessage(role=me.role, content=me.converted_prompt_text) for me in memory_entries]

