# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import NormalizerRequest, PromptConverterConfiguration
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import SubStringScorer


@pytest.fixture(scope="function")
def mock_target(patch_central_database) -> MockPromptTarget:
    return MockPromptTarget()


@patch(
    "pyrit.common.default_values.get_non_required_value",
    return_value='{"op_name": "dummy_op"}',
)
def test_init_orchestrator_global_memory_labels(get_non_required_value, mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
    assert orchestrator._global_memory_labels == {"op_name": "dummy_op"}


@pytest.mark.asyncio
async def test_run_attack_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    result = await orchestrator.run_attack_async(objective="Hello")
    assert mock_target.prompt_sent == ["Hello"]
    assert result.status == "unknown"  # No objective scorer
    assert result.conversation_id


@pytest.mark.asyncio
async def test_run_attacks_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    results = await orchestrator.run_attacks_async(objectives=["Hello", "World"])
    assert mock_target.prompt_sent == ["Hello", "World"]
    assert len(results) == 2
    assert all(r.status == "unknown" for r in results)  # No objective scorer
    assert len(set(r.conversation_id for r in results)) == 2  # Unique conversation IDs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prepended_conversation",
    [
        [],
        [
            PromptRequestResponse(request_pieces=[PromptRequestPiece(role="system", original_value="Hello")]),
        ],
    ],
)
async def test_run_attacks_with_prepended_conversation(mock_target: MockPromptTarget, prepended_conversation):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    results = await orchestrator.run_attacks_async(
        objectives=["Hello", "World"],
        prepended_conversations=[prepended_conversation, prepended_conversation]
    )
    assert mock_target.prompt_sent == ["Hello", "World"]
    assert len(results) == 2
    assert all(r.status == "unknown" for r in results)  # No objective scorer
    assert len(set(r.conversation_id for r in results)) == 2  # Unique conversation IDs


@pytest.mark.asyncio
async def test_run_attack_with_converter(mock_target: MockPromptTarget):
    converter = Base64Converter()
    converter_config = PromptConverterConfiguration.from_converters(converters=[converter])
    orchestrator = PromptSendingOrchestrator(
        objective_target=mock_target,
        request_converter_configurations=converter_config
    )

    result = await orchestrator.run_attack_async(objective="Hello")
    assert mock_target.prompt_sent == ["SGVsbG8="]
    assert result.status == "unknown"  # No objective scorer


@pytest.mark.asyncio
async def test_run_attack_with_multiple_converters(mock_target: MockPromptTarget):
    b64_converter = Base64Converter()
    join_converter = StringJoinConverter(join_value="_")
    converter_config = PromptConverterConfiguration.from_converters(converters=[b64_converter, join_converter])

    orchestrator = PromptSendingOrchestrator(
        objective_target=mock_target,
        request_converter_configurations=converter_config
    )

    result = await orchestrator.run_attack_async(objective="Hello")
    assert mock_target.prompt_sent == ["S_G_V_s_b_G_8_="]
    assert result.status == "unknown"  # No objective scorer


@pytest.mark.asyncio
async def test_run_attack_with_seed_prompt(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
    orchestrator._prompt_normalizer = AsyncMock()
    orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(return_value=None)

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(b"test")
        f.flush()

        group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=f.name,
                    data_type="image_path",
                )
            ]
        )

        result = await orchestrator.run_attack_async(
            objective="Elliciting harmful content through a SeedPrompt",
            seed_prompt=group
        )
        assert orchestrator._prompt_normalizer.send_prompt_async.called


@pytest.mark.asyncio
@pytest.mark.parametrize("num_conversations", [1, 10, 20])
async def test_run_attacks_with_scoring(mock_target: MockPromptTarget, num_conversations: int):
    scorer = SubStringScorer(
        substring="test",
        category="test",
    )

    scorer.score_async = AsyncMock()  # type: ignore

    orchestrator = PromptSendingOrchestrator(objective_target=mock_target, auxiliary_scorers=[scorer])
    orchestrator._prompt_normalizer = AsyncMock()

    request_pieces = []
    orchestrator_id = orchestrator.get_identifier()


    conversation_id = str(uuid.uuid4())
    request_pieces = [
                PromptRequestPiece(
                    role="user",
                    original_value=f"request_",
                    conversation_id=conversation_id,
                    orchestrator_identifier=orchestrator_id,
                ),
                PromptRequestPiece(
                    role="assistant",
                    original_value=f"response_",
                    conversation_id=conversation_id,
                    orchestrator_identifier=orchestrator_id,
                ),
            ]

    orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(
        return_value=PromptRequestResponse(request_pieces=request_pieces)
    )

    results = await orchestrator.run_attacks_async(
        objectives=[f"request_{n}" for n in range(num_conversations)]
    )
    
    assert orchestrator._prompt_normalizer.send_prompt_async.call_count == num_conversations
    assert scorer.score_async.call_count == num_conversations


@pytest.mark.asyncio
@pytest.mark.parametrize("num_prompts", [2, 20])
@pytest.mark.parametrize("max_rpm", [30])
async def test_max_requests_per_minute_delay(patch_central_database, num_prompts: int, max_rpm: int):
    mock_target = MockPromptTarget(rpm=max_rpm)
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target, batch_size=1)

    objectives = ["test"] * num_prompts

    start = time.time()
    await orchestrator.run_attacks_async(objectives=objectives)
    end = time.time()

    assert (end - start) > (60 / max_rpm * num_prompts)


def test_orchestrator_sets_target_memory(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
    assert orchestrator._memory is mock_target._memory


def test_send_prompt_to_identifier(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    d = orchestrator.get_identifier()
    assert d["id"]
    assert d["__type__"] == "PromptSendingOrchestrator"
    assert d["__module__"] == "pyrit.orchestrator.single_turn.prompt_sending_orchestrator"


def test_orchestrator_get_memory(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    request = PromptRequestPiece(
        role="user",
        original_value="test",
        orchestrator_identifier=orchestrator.get_identifier(),
    ).to_prompt_request_response()

    orchestrator._memory.add_request_response_to_memory(request=request)

    entries = orchestrator.get_memory()
    assert entries
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_run_attacks_with_env_local_memory_labels(mock_target: MockPromptTarget):
    with patch(
        "os.environ.get",
        side_effect=lambda key, default=None: '{"op_name": "dummy_op"}' if key == "GLOBAL_MEMORY_LABELS" else default,
    ):
        orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
        results = await orchestrator.run_attacks_async(objectives=["hello"])
        assert mock_target.prompt_sent == ["hello"]

        expected_labels = {"op_name": "dummy_op"}
        entries = orchestrator.get_memory()
        assert len(entries) == 2
        assert entries[0].labels == expected_labels
        assert entries[1].labels == expected_labels


@pytest.mark.asyncio
async def test_run_attacks_with_memory_labels(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
    new_labels = {"op_name": "op1", "username": "name1"}
    results = await orchestrator.run_attacks_async(
        objectives=["hello"],
        memory_labels=new_labels
    )
    assert mock_target.prompt_sent == ["hello"]

    expected_labels = {"op_name": "op1", "username": "name1"}
    entries = orchestrator.get_memory()
    assert len(entries) == 2
    assert entries[0].labels == expected_labels
    assert entries[1].labels == expected_labels


@pytest.mark.asyncio
async def test_run_attacks_combine_memory_labels(mock_target: MockPromptTarget):
    with patch(
        "os.environ.get",
        side_effect=lambda key, default=None: '{"op_name": "dummy_op"}' if key == "GLOBAL_MEMORY_LABELS" else default,
    ):
        orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
        new_labels = {"op_name": "op2", "username": "dummy_name"}
        results = await orchestrator.run_attacks_async(
            objectives=["hello"],
            memory_labels=new_labels
        )
        assert mock_target.prompt_sent == ["hello"]

        expected_labels = {"op_name": "op2", "username": "dummy_name"}
        entries = orchestrator.get_memory()
        assert len(entries) == 2
        assert entries[0].labels == expected_labels


@pytest.mark.asyncio
async def test_orchestrator_get_score_memory(mock_target: MockPromptTarget):
    scorer = AsyncMock()
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target, auxiliary_scorers=[scorer])

    request = PromptRequestPiece(
        role="user",
        original_value="dummytest",
        orchestrator_identifier=orchestrator.get_identifier(),
    )

    score = Score(
        score_type="float_scale",
        score_value=str(1),
        score_value_description=None,
        score_category="mock",
        score_metadata=None,
        score_rationale=None,
        scorer_class_identifier=orchestrator.get_identifier(),
        prompt_request_response_id=request.id,
    )

    orchestrator._memory.add_request_pieces_to_memory(request_pieces=[request])
    orchestrator._memory.add_scores_to_memory(scores=[score])
    with patch.object(orchestrator._memory, "get_prompt_request_pieces", return_value=[request]):
        scores = orchestrator.get_score_memory()
        assert len(scores) == 1
        assert scores[0].prompt_request_response_id == request.id


@pytest.mark.parametrize("orchestrator_count", [10, 100])
def test_orchestrator_unique_id(orchestrator_count: int):
    orchestrator_ids = set()
    duplicate_found = False
    for n in range(orchestrator_count):
        id = PromptSendingOrchestrator(objective_target=MagicMock()).get_identifier()["id"]

        if id in orchestrator_ids:
            duplicate_found = True
            break

        orchestrator_ids.add(id)

    assert not duplicate_found
