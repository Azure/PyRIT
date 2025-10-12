# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations

from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import BatchScorer


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestResponse]:
    return get_sample_conversations()


@pytest.mark.usefixtures("patch_central_database")
class TestBatchScorerInitialization:
    """Test BatchScorer initialization scenarios."""

    def test_init_with_default_batch_size(self) -> None:
        """Test initialization with default batch size."""
        with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
            batch_scorer = BatchScorer()
            assert batch_scorer._batch_size == 10

    def test_init_with_custom_batch_size(self) -> None:
        """Test initialization with custom batch size."""
        with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
            batch_scorer = BatchScorer(batch_size=5)
            assert batch_scorer._batch_size == 5

    def test_init_memory_initialization(self) -> None:
        """Test that memory is properly initialized from CentralMemory."""
        mock_memory = MagicMock()
        with patch.object(CentralMemory, "get_memory_instance", return_value=mock_memory) as mock_get_memory:
            batch_scorer = BatchScorer()

            mock_get_memory.assert_called_once()
            assert batch_scorer._memory == mock_memory


@pytest.mark.usefixtures("patch_central_database")
class TestBatchScorerScoreResponsesByFilters:
    """Test score_responses_by_filters_async method functionality."""

    @pytest.mark.asyncio
    async def test_score_responses_by_filters_basic_functionality(
        self, sample_conversations: MutableSequence[PromptRequestResponse]
    ) -> None:
        """Test basic scoring functionality with filters."""
        memory = MagicMock()
        memory.get_prompt_request_pieces.return_value = [sample_conversations[1].request_pieces[0]]

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            scorer = MagicMock()
            test_score = MagicMock()
            scorer.score_prompts_batch_async = AsyncMock(return_value=[test_score])

            batch_scorer = BatchScorer()

            scores = await batch_scorer.score_responses_by_filters_async(scorer=scorer, attack_id=str(uuid.uuid4()))

            memory.get_prompt_request_pieces.assert_called_once()
            scorer.score_prompts_batch_async.assert_called_once()
            assert scores[0] == test_score

    @pytest.mark.asyncio
    async def test_score_responses_by_filters_with_all_parameters(
        self, sample_conversations: MutableSequence[PromptRequestResponse]
    ) -> None:
        """Test scoring with all filter parameters."""
        memory = MagicMock()
        memory.get_prompt_request_pieces.return_value = [sample_conversations[1].request_pieces[0]]

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            scorer = MagicMock()
            scorer.score_prompts_batch_async = AsyncMock(return_value=[])

            batch_scorer = BatchScorer()

            test_attack_id = str(uuid.uuid4())
            test_conversation_id = str(uuid.uuid4())
            test_prompt_ids = ["id1", "id2"]
            test_labels = {"test": "value"}
            test_data_type = "text"
            test_not_data_type = "image"

            await batch_scorer.score_responses_by_filters_async(
                scorer=scorer,
                attack_id=test_attack_id,
                conversation_id=test_conversation_id,
                prompt_ids=test_prompt_ids,
                labels=test_labels,
                data_type=test_data_type,
                not_data_type=test_not_data_type,
            )

            # Should call memory with all parameters including None for unspecified ones
            memory.get_prompt_request_pieces.assert_called_once_with(
                attack_id=test_attack_id,
                conversation_id=test_conversation_id,
                prompt_ids=test_prompt_ids,
                labels=test_labels,
                sent_after=None,
                sent_before=None,
                original_values=None,
                converted_values=None,
                data_type=test_data_type,
                not_data_type=test_not_data_type,
                converted_value_sha256=None,
            )

    @pytest.mark.asyncio
    async def test_score_responses_by_filters_raises_error_no_matching_filters(self) -> None:
        """Test that ValueError is raised when no entries match filters."""
        memory = MagicMock()
        memory.get_prompt_request_pieces.return_value = []

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            batch_scorer = BatchScorer()

            with pytest.raises(ValueError, match="No entries match the provided filters. Please check your filters."):
                await batch_scorer.score_responses_by_filters_async(
                    scorer=MagicMock(),
                    labels={"op_name": "nonexistent_op", "user_name": "nonexistent_user"},
                )


@pytest.mark.usefixtures("patch_central_database")
class TestBatchScorerUtilityMethods:
    """Test utility methods of BatchScorer."""

    def test_remove_duplicates(self) -> None:
        """Test removal of duplicate prompt request pieces."""
        prompt_id1 = uuid.uuid4()
        prompt_id2 = uuid.uuid4()

        with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
            batch_scorer = BatchScorer()

            pieces = [
                PromptRequestPiece(
                    id=prompt_id1,
                    role="user",
                    original_value="original prompt text",
                    converted_value="Hello, how are you?",
                    sequence=0,
                ),
                PromptRequestPiece(
                    id=prompt_id2,
                    role="assistant",
                    original_value="original prompt text",
                    converted_value="I'm fine, thank you!",
                    sequence=1,
                ),
                PromptRequestPiece(
                    role="user",
                    original_value="original prompt text",
                    converted_value="Hello, how are you?",
                    sequence=0,
                    original_prompt_id=prompt_id1,
                ),
                PromptRequestPiece(
                    role="assistant",
                    original_value="original prompt text",
                    converted_value="I'm fine, thank you!",
                    sequence=1,
                    original_prompt_id=prompt_id2,
                ),
            ]

            orig_pieces = batch_scorer._remove_duplicates(pieces)

            assert len(orig_pieces) == 2
            for piece in orig_pieces:
                assert piece.id in [prompt_id1, prompt_id2]
                assert piece.id == piece.original_prompt_id


@pytest.mark.usefixtures("patch_central_database")
class TestBatchScorerErrorHandling:
    """Test error handling scenarios for BatchScorer."""

    @pytest.mark.asyncio
    async def test_score_responses_by_filters_no_filters_provided(
        self, sample_conversations: MutableSequence[PromptRequestResponse]
    ) -> None:
        """Test scoring when no filters are provided."""
        memory = MagicMock()
        memory.get_prompt_request_pieces.return_value = [sample_conversations[1].request_pieces[0]]

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            scorer = MagicMock()
            scorer.score_prompts_batch_async = AsyncMock(return_value=[])

            batch_scorer = BatchScorer()

            await batch_scorer.score_responses_by_filters_async(scorer=scorer)

            # Should call memory with all None parameters
            memory.get_prompt_request_pieces.assert_called_once_with(
                attack_id=None,
                conversation_id=None,
                prompt_ids=None,
                labels=None,
                sent_after=None,
                sent_before=None,
                original_values=None,
                converted_values=None,
                data_type=None,
                not_data_type=None,
                converted_value_sha256=None,
            )

    @pytest.mark.asyncio
    async def test_score_responses_by_filters_handles_multiple_conversations(self) -> None:
        """Test that scoring handles pieces from multiple conversations correctly."""
        memory = MagicMock()

        # Create pieces from different conversations
        pieces = [
            PromptRequestPiece(
                role="user",
                conversation_id="conv1",
                sequence=0,
                original_value="Conv1 message",
            ),
            PromptRequestPiece(
                role="assistant",
                conversation_id="conv1",
                sequence=1,
                original_value="Conv1 response",
            ),
            PromptRequestPiece(
                role="user",
                conversation_id="conv2",
                sequence=0,
                original_value="Conv2 message",
            ),
            PromptRequestPiece(
                role="assistant",
                conversation_id="conv2",
                sequence=1,
                original_value="Conv2 response",
            ),
        ]

        memory.get_prompt_request_pieces.return_value = pieces

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            scorer = MagicMock()
            test_scores = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
            scorer.score_prompts_batch_async = AsyncMock(return_value=test_scores)

            batch_scorer = BatchScorer()

            scores = await batch_scorer.score_responses_by_filters_async(
                scorer=scorer,
                attack_id=str(uuid.uuid4()),
            )

            # Should successfully group by conversation and sequence
            scorer.score_prompts_batch_async.assert_called_once()
            call_args = scorer.score_prompts_batch_async.call_args
            request_responses = call_args.kwargs["request_responses"]

            # Should have 4 groups: 2 sequences from conv1, 2 sequences from conv2
            assert len(request_responses) == 4
            assert len(scores) == 4

    @pytest.mark.asyncio
    async def test_score_responses_by_filters_groups_by_sequence_within_conversation(self) -> None:
        """Test that pieces are properly grouped by sequence within each conversation."""
        memory = MagicMock()

        # Create multiple pieces in the same sequence
        pieces = [
            PromptRequestPiece(
                role="system",
                conversation_id="conv1",
                sequence=0,
                original_value="System prompt",
            ),
            PromptRequestPiece(
                role="user",
                conversation_id="conv1",
                sequence=1,
                original_value="User message 1",
            ),
            PromptRequestPiece(
                role="user",
                conversation_id="conv1",
                sequence=1,
                original_value="User message 2",
            ),
            PromptRequestPiece(
                role="assistant",
                conversation_id="conv1",
                sequence=2,
                original_value="Assistant response",
            ),
        ]

        memory.get_prompt_request_pieces.return_value = pieces

        with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
            scorer = MagicMock()
            scorer.score_prompts_batch_async = AsyncMock(return_value=[MagicMock(), MagicMock()])

            batch_scorer = BatchScorer()

            await batch_scorer.score_responses_by_filters_async(scorer=scorer, conversation_id="conv1")

            # Verify grouping
            call_args = scorer.score_prompts_batch_async.call_args
            request_responses = call_args.kwargs["request_responses"]

            # Should have 3 groups: sequence 0 (with 1 pieces) and sequence 1 (with 2 pieces)
            assert len(request_responses) == 3
            assert len(request_responses[0].request_pieces) == 1
            assert len(request_responses[1].request_pieces) == 2
            assert len(request_responses[2].request_pieces) == 1
