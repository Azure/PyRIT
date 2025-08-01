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
    - Database lifecycle, Embedding management, Abstract database operations, Utility methods:
    ## Scores (MemoryScoresMixin)
    ## Prompts & Conversations (MemoryPromptsMixin)
    ## Seed Prompts (MemorySeedPromptsMixin)
    ## Attack Results (MemoryAttackResultsMixin)
    ## Export & Utilities (MemoryExportMixin)
    """

    pass
