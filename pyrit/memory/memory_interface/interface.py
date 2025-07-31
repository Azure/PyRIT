# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
MemoryInterface using mixins to organize functionality into logical groups.

This approach maintains the single class interface while improving code organization.
The functionality is grouped into:
- Infrastructure: Core database operations, lifecycle management, embedding
- Scores: Score-related operations
- Prompts: Prompt and conversation management  
- Seed Prompts: Seed prompt operations
- Attack Results: Attack result operations
- Export: Export and utility functions
"""

import abc

from pyrit.memory.memory_interface.attack_results_mixin import MemoryAttackResultsMixin
from pyrit.memory.memory_interface.export_mixin import MemoryExportMixin
from pyrit.memory.memory_interface.infrastructure_mixin import MemoryInfrastructureMixin
from pyrit.memory.memory_models import PromptRequestPiece
from pyrit.memory.memory_interface.prompts_mixin import MemoryPromptsMixin
from pyrit.memory.memory_interface.scores_mixin import MemoryScoresMixin
from pyrit.memory.memory_interface.seed_prompts_mixin import MemorySeedPromptsMixin


class MemoryInterface(
    MemoryInfrastructureMixin,
    MemoryScoresMixin,
    MemoryPromptsMixin,
    MemorySeedPromptsMixin,
    MemoryAttackResultsMixin,
    MemoryExportMixin,
    abc.ABC,
):
    """Abstract interface for conversation memory storage systems.

    This interface defines the contract for storing and retrieving chat messages
    and conversation history. Implementations can use different storage backends
    such as files, databases, or cloud storage services.

    The functionality is organized into logical groups through mixins:

    ## Infrastructure (MemoryInfrastructureMixin)
    - Database lifecycle: __init__, cleanup, dispose_engine
    - Embedding management: enable_embedding, disable_embedding, get_all_embeddings
    - Abstract database operations: _init_storage_io, _query_entries, _insert_entry, _insert_entries, _update_entries
    - Utility methods: _add_list_conditions, print_schema

    ## Scores (MemoryScoresMixin)  
    - add_scores_to_memory
    - get_scores_by_prompt_ids
    - get_scores_by_orchestrator_id
    - get_scores_by_memory_labels

    ## Prompts & Conversations (MemoryPromptsMixin)
    - get_conversation
    - get_prompt_request_pieces (main query method)
    - duplicate_conversation, duplicate_conversation_excluding_last_turn
    - add_request_response_to_memory, _update_sequence
    - update_prompt_entries_by_conversation_id, update_labels_by_conversation_id, update_prompt_metadata_by_conversation_id
    - get_chat_messages_with_conversation_id

    ## Seed Prompts (MemorySeedPromptsMixin)
    - get_seed_prompts
    - add_seed_prompts_to_memory_async, _serialize_seed_prompt_value
    - get_seed_prompt_dataset_names
    - add_seed_prompt_groups_to_memory, get_seed_prompt_groups

    ## Attack Results (MemoryAttackResultsMixin)
    - add_attack_results_to_memory
    - get_attack_results

    ## Export & Utilities (MemoryExportMixin)
    - export_conversations
    """

    pass
