# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.models.filter_criteria import PromptFilterCriteria
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
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
        objectives=["Hello", "World"], prepended_conversations=[prepended_conversation, prepended_conversation]
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
        objective_target=mock_target, request_converter_configurations=converter_config
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
        objective_target=mock_target, request_converter_configurations=converter_config
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

        await orchestrator.run_attack_async(
            objective="Elliciting harmful content through a SeedPrompt", seed_prompt=group
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

    orchestrator_id = orchestrator.get_identifier()
    conversation_id = str(uuid.uuid4())
    request_pieces = [
        PromptRequestPiece(
            role="user",
            original_value="request_",
            conversation_id=conversation_id,
            orchestrator_identifier=orchestrator_id,
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="response_",
            conversation_id=conversation_id,
            orchestrator_identifier=orchestrator_id,
        ),
    ]

    orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(
        return_value=PromptRequestResponse(request_pieces=request_pieces)
    )

    await orchestrator.run_attacks_async(objectives=[f"request_{n}" for n in range(num_conversations)])

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
        await orchestrator.run_attacks_async(objectives=["hello"])
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
    await orchestrator.run_attacks_async(objectives=["hello"], memory_labels=new_labels)
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
        await orchestrator.run_attacks_async(objectives=["hello"], memory_labels=new_labels)
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
    for _ in range(orchestrator_count):
        id = PromptSendingOrchestrator(objective_target=MagicMock()).get_identifier()["id"]

        if id in orchestrator_ids:
            duplicate_found = True
            break

        orchestrator_ids.add(id)

    assert not duplicate_found


@pytest.mark.asyncio
async def test_run_attack_with_objective_scorer(mock_target: MockPromptTarget):
    scorer = SubStringScorer(substring="success", category="test")
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target, objective_scorer=scorer)

    conversation_id = str(uuid.uuid4())
    orchestrator_id = orchestrator.get_identifier()

    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="This is a success message",
                conversation_id=conversation_id,
                orchestrator_identifier=orchestrator_id,
            )
        ]
    )

    with patch.object(mock_target, "send_prompt_async", return_value=response):
        result = await orchestrator.run_attack_async(objective="This is a success message")
        assert result.status == "success"
        assert result.objective_score is not None


@pytest.mark.asyncio
async def test_run_attack_with_objective_scorer_failure(mock_target: MockPromptTarget):
    # Test with a true/false scorer that fails
    scorer = SubStringScorer(substring="success", category="test")
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target, objective_scorer=scorer)

    result = await orchestrator.run_attack_async(objective="This is a failure message")
    assert result.status == "failure"
    assert result.objective_score is not None


@pytest.mark.asyncio
async def test_run_attack_with_invalid_objective_scorer(mock_target: MockPromptTarget):
    # Test with a non-true/false scorer
    scorer = MagicMock()
    scorer.scorer_type = "invalid"

    with pytest.raises(ValueError, match="Objective scorer must be a true/false scorer"):
        PromptSendingOrchestrator(objective_target=mock_target, objective_scorer=scorer)


@pytest.mark.asyncio
async def test_run_attack_with_retries(mock_target: MockPromptTarget):
    # Create a mock scorer that fails twice then succeeds
    mock_scorer = AsyncMock()
    mock_scorer.score_async.side_effect = [
        [
            Score(
                score_type="true_false",
                score_value="false",
                score_category="test",
                score_value_description=None,
                score_rationale=None,
                score_metadata=None,
                prompt_request_response_id="test_id",
            )
        ],
        [
            Score(
                score_type="true_false",
                score_value="false",
                score_category="test",
                score_value_description=None,
                score_rationale=None,
                score_metadata=None,
                prompt_request_response_id="test_id",
            )
        ],
        [
            Score(
                score_type="true_false",
                score_value="true",
                score_category="test",
                score_value_description=None,
                score_rationale=None,
                score_metadata=None,
                prompt_request_response_id="test_id",
            )
        ],
    ]
    mock_scorer.scorer_type = "true_false"

    orchestrator = PromptSendingOrchestrator(
        objective_target=mock_target, objective_scorer=mock_scorer, retries_on_objective_failure=2
    )

    # Mock the normalizer to return a simple response
    conversation_id = str(uuid.uuid4())
    orchestrator_id = orchestrator.get_identifier()
    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="test response",
                conversation_id=conversation_id,
                orchestrator_identifier=orchestrator_id,
            )
        ]
    )

    with patch.object(
        orchestrator._prompt_normalizer, "send_prompt_async", new_callable=AsyncMock, return_value=response
    ) as mock_send_prompt:
        result = await orchestrator.run_attack_async(objective="test prompt")
        assert result.status == "success"
        assert mock_send_prompt.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_run_attack_with_skip_criteria(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    # Set skip criteria to skip prompts that match in memory
    skip_criteria = PromptFilterCriteria(labels={"test": "value"}, not_data_type="error")
    orchestrator.set_skip_criteria(skip_criteria=skip_criteria)

    # First run should succeed
    result1 = await orchestrator.run_attack_async(objective="test prompt", memory_labels={"test": "value"})
    assert result1 is not None

    with patch.object(orchestrator._prompt_normalizer, "send_prompt_async", return_value=None):
        # Second run with same prompt should be skipped
        result2 = await orchestrator.run_attack_async(objective="test prompt", memory_labels={"test": "value"})
        assert result2 is None


@pytest.mark.asyncio
async def test_run_attack_with_response_converter(mock_target: MockPromptTarget):
    response_converter = Base64Converter()
    converter_config = PromptConverterConfiguration.from_converters(converters=[response_converter])

    orchestrator = PromptSendingOrchestrator(
        objective_target=mock_target, response_converter_configurations=converter_config  # Not a list
    )

    conversation_id = str(uuid.uuid4())
    orchestrator_id = orchestrator.get_identifier()

    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="test response",
                conversation_id=conversation_id,
                orchestrator_identifier=orchestrator_id,
            )
        ]
    )

    with patch.object(mock_target, "send_prompt_async", return_value=response):
        result = await orchestrator.run_attack_async(objective="test prompt")
        assert result is not None
        conversation = orchestrator._memory.get_prompt_request_pieces(conversation_id=conversation_id, role="assistant")
        assert any(piece.converted_value != piece.original_value for piece in conversation)


@pytest.mark.asyncio
async def test_run_attack_with_memory_labels_override(mock_target: MockPromptTarget):
    # Test that memory labels can override global labels
    with patch("os.environ.get", return_value='{"op_name": "global_op"}'):
        orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
        await orchestrator.run_attack_async(objective="test prompt", memory_labels={"op_name": "local_op"})

        entries = orchestrator.get_memory()
        assert entries[0].labels["op_name"] == "local_op"


@pytest.mark.asyncio
async def test_run_attack_with_prepended_conversation_error(patch_central_database):
    orchestrator = PromptSendingOrchestrator(objective_target=MagicMock())

    prepended_conversation = [
        PromptRequestResponse(request_pieces=[PromptRequestPiece(role="system", original_value="test")])
    ]

    with pytest.raises(ValueError, match="Prepended conversation can only be used with a PromptChatTarget"):
        await orchestrator.run_attack_async(objective="test prompt", prepended_conversation=prepended_conversation)


@pytest.mark.asyncio
async def test_run_attack_with_multiple_auxiliary_scorers(mock_target: MockPromptTarget):
    # Create mock scorers
    scorer1 = AsyncMock()
    scorer2 = AsyncMock()

    orchestrator = PromptSendingOrchestrator(objective_target=mock_target, auxiliary_scorers=[scorer1, scorer2])

    conversation_id = str(uuid.uuid4())
    orchestrator_id = orchestrator.get_identifier()

    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="test response",
                conversation_id=conversation_id,
                orchestrator_identifier=orchestrator_id,
            )
        ]
    )

    with patch.object(mock_target, "send_prompt_async", return_value=response):
        result = await orchestrator.run_attack_async(objective="test prompt")
        assert result is not None

        assert scorer1.score_async.call_count == 1
        assert scorer2.score_async.call_count == 1


@pytest.mark.asyncio
async def test_run_attack_with_seed_prompt_and_objective(mock_target: MockPromptTarget):
    # Test using both seed prompt and objective
    seed_prompt = SeedPromptGroup(prompts=[SeedPrompt(value="seed prompt", data_type="text")])

    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)
    result = await orchestrator.run_attack_async(objective="test objective", seed_prompt=seed_prompt)

    assert result is not None
    assert result.objective == "test objective"
    assert mock_target.prompt_sent[0] == "seed prompt"


@pytest.mark.asyncio
async def test_run_attacks_with_mismatched_seed_prompts(mock_target: MockPromptTarget):
    # Test error when seed prompts don't match objectives
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    with pytest.raises(ValueError, match="Number of seed prompts must match number of objectives"):
        await orchestrator.run_attacks_async(
            objectives=["obj1", "obj2"],
            seed_prompts=[SeedPromptGroup(prompts=[SeedPrompt(value="prompt1", data_type="text")])],
        )


@pytest.mark.asyncio
async def test_run_attacks_with_mismatched_prepended_conversations(mock_target: MockPromptTarget):
    # Test error when prepended conversations don't match objectives
    orchestrator = PromptSendingOrchestrator(objective_target=mock_target)

    with pytest.raises(ValueError, match="Number of prepended conversations must match number of objectives"):
        await orchestrator.run_attacks_async(
            objectives=["obj1", "obj2"],
            prepended_conversations=[
                [PromptRequestResponse(request_pieces=[PromptRequestPiece(role="system", original_value="test")])]
            ],
        )
