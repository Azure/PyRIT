import pytest
import uuid

from unittest.mock import MagicMock
from pyrit.orchestrator.multi_turn.red_teaming_orchestrator import RedTeamingOrchestrator


@pytest.fixture
def orchestrator():
    objective_scorer = MagicMock()
    objective_scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        objective_target=MagicMock(),
        adversarial_chat=MagicMock(),
        max_turns=2,
        objective_scorer=objective_scorer,
    )

    orchestrator._memory = MagicMock()
    return orchestrator


def test_prepare_conversation_no_prepended_conversation(orchestrator):
    orchestrator._prepended_conversation = None
    new_conversation_id = str(uuid.uuid4())
    num_turns = orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)
    assert num_turns == 0
    orchestrator._memory.add_request_response_to_memory.assert_not_called()


def test_prepare_conversation_with_prepended_conversation(orchestrator):
    request_piece = MagicMock()
    request_piece.role = "assistant"
    id = uuid.uuid4()
    request_piece.id = id
    request = MagicMock()
    request.request_pieces = [request_piece]

    orchestrator._prepended_conversation = [request]
    new_conversation_id = str(uuid.uuid4())
    num_turns = orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)

    assert num_turns == 1
    assert request_piece.conversation_id == new_conversation_id
    assert request_piece.id != id
    orchestrator._memory.add_request_response_to_memory.assert_called_once_with(request=request)


def test_prepare_conversation_exceeds_max_turns(orchestrator):
    request_piece = MagicMock()
    request_piece.role = "assistant"
    request = MagicMock()
    request.request_pieces = [request_piece]
    orchestrator._prepended_conversation = [request] * (orchestrator._max_turns)
    new_conversation_id = str(uuid.uuid4())

    with pytest.raises(ValueError):
        orchestrator._prepare_conversation(new_conversation_id=new_conversation_id)
