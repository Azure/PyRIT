# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
import uuid
from unittest.mock import MagicMock, Mock, AsyncMock, ANY
from unittest.mock import patch

import pytest

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse, Score
from pyrit.orchestrator import PAIROrchestrator
from pyrit.orchestrator.multi_turn.pair_orchestrator import PromptRequestPiece
from pyrit.orchestrator.multi_turn.tree_of_attacks_with_pruning_orchestrator import TreeOfAttacksWithPruningOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import Scorer
from pyrit.memory import CentralMemory
from tests.mocks import get_memory_interface



def test_init():
    orchestrator = PAIROrchestrator(
        objective_target=MagicMock(),
        adversarial_chat=MagicMock(),
        scoring_target=MagicMock(),
    )

    assert orchestrator._attack_width == 1
    assert orchestrator._attack_branching_factor == 1
    assert orchestrator._adversarial_chat_system_seed_prompt.name == "pair_attacker_system_prompt"

def test_pair_is_tap():
    assert issubclass(PAIROrchestrator, TreeOfAttacksWithPruningOrchestrator)

