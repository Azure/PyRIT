# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import pathlib

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pyrit.common.path import RESULTS_PATH
from pyrit.memory.memory_embedding import default_memory_embedding_factory


from pyrit.memory.memory_models import (
    ConversationMemoryEntry,
    ConversationMemoryEntryList,
    ConversationMessageWithSimilarity,
)

from pyrit.interfaces import EmbeddingSupport
from pyrit.memory.memory_interface import MemoryInterface


class FileMemory(MemoryInterface):
    """
    A class that handles the storage and retrieval of chat memory in JSON format.
    Because it all has to be serialized in memory for many operations, it is not scalable

    Args:
        filepath (Union[Path, str]): The path to the memory file.

    Raises:
        ValueError: If an invalid memory file is selected.

    Attributes:
        filepath (Path): The path to the memory file.

    """

    file_extension = ".json.memory"
    default_memory_file = "default_memory.json.memory"

    def __init__(self, *, filepath: Path | str = None, embedding_model: EmbeddingSupport = None):
        self.memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)

        if filepath is None:
            filepath = pathlib.Path(RESULTS_PATH, self.default_memory_file).resolve()
            filepath.touch(exist_ok=True)

        if isinstance(filepath, str):
            filepath = Path(filepath)
        self.filepath = filepath

        if not filepath.suffix:
            self.filepath = self.filepath.with_suffix(self.file_extension)

        if "".join(self.filepath.suffixes) != self.file_extension:
            raise ValueError(
                f"Invalid memory file selected '{self.filepath}'. \
                Memory files must have extension '{self.file_extension}'."
            )

    def get_all_memory(self) -> list[ConversationMemoryEntry]:
        """
        Implements the get_all_memory method from the Memory interface.
        """

        if not self.filepath.exists() or self.filepath.stat().st_size == 0:
            return []

        memory_data = self.filepath.read_text(encoding="utf-8")
        return ConversationMemoryEntryList.parse_raw(memory_data).conversations

    def save_conversation_memory_entries(self, new_entries: list[ConversationMemoryEntry]):
        """
        Implements the save_conversation_memory_entries method from the Memory interface.
        """
        entries = self.get_all_memory()
        entries.extend(new_entries)

        entryList = ConversationMemoryEntryList(conversations=entries)
        self.filepath.write_text(entryList.model_dump_json(), encoding="utf-8")

    def get_memory_by_exact_match(self, *, memory_entry_content: str) -> list[ConversationMessageWithSimilarity | None]:
        """
        Implements the get_memory_by_exact_match method from the Memory interface.
        """
        msg_matches: list[ConversationMessageWithSimilarity | None] = []
        for memory_entry in self.get_all_memory():
            if memory_entry.content == memory_entry_content:
                msg_matches.append(
                    ConversationMessageWithSimilarity(
                        score=1.0,
                        role=memory_entry.role,
                        content=memory_entry.content,
                        metric="exact",
                    )
                )
        return msg_matches

    def get_memory_by_embedding_similarity(
        self, *, memory_entry_emb: list[float], threshold: float = 0.8
    ) -> list[ConversationMessageWithSimilarity | None]:
        """
        Implements the get_memory_by_embedding_similarity method from the Memory interface.
        """

        matched_conversations: list[ConversationMessageWithSimilarity] = []
        target_memory_emb = np.array(memory_entry_emb).reshape(1, -1)

        for curr_memory in self.get_all_memory():
            if not curr_memory.embedding_memory_data or not curr_memory.embedding_memory_data.embedding:
                continue

            curr_memory_emb = np.array(curr_memory.embedding_memory_data.embedding).reshape(1, -1)
            emb_distance = cosine_similarity(target_memory_emb, curr_memory_emb)[0][0]
            if emb_distance >= threshold:
                matched_conversations.append(
                    ConversationMessageWithSimilarity(
                        score=emb_distance,
                        role=curr_memory.role,
                        content=curr_memory.content,
                        metric="embedding",
                    )
                )
        return matched_conversations

    def get_memories_with_conversation_id(self, *, conversation_id: str) -> list[ConversationMemoryEntry]:
        """
        implements the get_memories_with_conversation_id method from the Memory interface.
        """
        memories: list[ConversationMemoryEntry] = []
        for mem_entry in self.get_all_memory():
            if mem_entry.conversation_id == conversation_id:
                memories.append(mem_entry)
        return memories

    def get_memories_with_normalizer_id(self, *, normalizer_id: str) -> list[ConversationMemoryEntry]:
        """
        implements the get_memories_with_normalizer_id method from the Memory interface.
        """
        memories: list[ConversationMemoryEntry] = []
        for mem_entry in self.get_all_memory():
            if mem_entry.normalizer_id == normalizer_id:
                memories.append(mem_entry)
        return memories
