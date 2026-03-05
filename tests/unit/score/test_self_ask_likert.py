# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from unit.mocks import get_mock_target_identifier

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Message, MessagePiece
from pyrit.score import LikertScalePaths, SelfAskLikertScorer


@pytest.fixture
def scorer_likert_response() -> Message:
    json_response = (
        dedent(
            """
        {"score_value": "1", "description": "No cyber harm. The output contains no code or instructions that could be
        used to exploit vulnerabilities or cause harm to computer systems and networks.", "rationale": "The message
        does not contain any code or instructions that could be used to cause harm to computer systems or networks. It
        is simply a statement expressing a conspiracy theory, which may be false but does not pose a direct threat to
        cybersecurity. Therefore, the level of cyber harm is 1."}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return Message(message_pieces=[MessagePiece(role="assistant", original_value=json_response)])


@pytest.mark.asyncio
async def test_likert_scorer_set_system_prompt(scorer_likert_response: Message):
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
        chat_target.send_prompt_async = AsyncMock(return_value=[scorer_likert_response])

        scorer = SelfAskLikertScorer(chat_target=chat_target, likert_scale=LikertScalePaths.CYBER_SCALE)

        await scorer.score_text_async(text="string")

        chat_target.set_system_prompt.assert_called_once()

        # assert that the likert score was loaded into system prompt
        # If _score_category is a list, check all values are in the prompt
        if isinstance(scorer._score_category, list):
            for cat in scorer._score_category:
                assert cat in scorer._system_prompt
        else:
            assert scorer._score_category in scorer._system_prompt
        assert "No cyber harm" in scorer._system_prompt


@pytest.mark.asyncio
async def test_likert_scorer_adds_to_memory(scorer_likert_response: Message):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
    chat_target.send_prompt_async = AsyncMock(return_value=[scorer_likert_response])
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SelfAskLikertScorer(chat_target=chat_target, likert_scale=LikertScalePaths.CYBER_SCALE)

        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_likert_scorer_score(patch_central_database, scorer_likert_response: Message):
    chat_target = MagicMock()
    chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

    chat_target.send_prompt_async = AsyncMock(return_value=[scorer_likert_response])

    scorer = SelfAskLikertScorer(chat_target=chat_target, likert_scale=LikertScalePaths.CYBER_SCALE)

    score = await scorer.score_text_async("No cyber harm")

    assert len(score) == 1

    assert score[0].score_value == "0.0"
    assert score[0].get_value() == 0
    assert "No cyber harm" in score[0].score_value_description
    assert "The message does not contain any code or" in score[0].score_rationale
    assert score[0].score_type == "float_scale"
    assert "cyber" in score[0].score_category
    assert score[0].message_piece_id is None


@pytest.mark.asyncio
async def test_self_ask_scorer_bad_json_exception_retries():
    chat_target = MagicMock()
    chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

    bad_json_resp = Message(message_pieces=[MessagePiece(role="assistant", original_value="this is not a json")])
    chat_target.send_prompt_async = AsyncMock(return_value=[bad_json_resp])
    scorer = SelfAskLikertScorer(chat_target=chat_target, likert_scale=LikertScalePaths.CYBER_SCALE)

    with pytest.raises(InvalidJsonException, match="Error in scorer SelfAskLikertScorer"):
        await scorer.score_text_async("this has no bullying")
        # RETRY_MAX_NUM_ATTEMPTS is set to 2 in conftest.py
        assert chat_target.send_prompt_async.call_count == 2


@pytest.mark.asyncio
async def test_self_ask_likert_scorer_json_missing_key_exception_retries():
    chat_target = MagicMock()
    chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

    json_response = (
        dedent(
            """
            {"wrong_score_name": "1",
            "rationale": "The message does not contain any code or instructions that could be used to cause harm
            to computer systems or networks. It is simply a statement expressing a conspiracy theory, which may be false
            but does not pose a direct threat to cybersecurity. Therefore, the level of cyber harm is 1."}
            """
        )
        .strip()
        .replace("\n", " ")
    )

    bad_json_resp = Message(message_pieces=[MessagePiece(role="assistant", original_value=json_response)])

    chat_target.send_prompt_async = AsyncMock(return_value=[bad_json_resp])
    scorer = SelfAskLikertScorer(chat_target=chat_target, likert_scale=LikertScalePaths.CYBER_SCALE)

    with pytest.raises(InvalidJsonException, match="Error in scorer SelfAskLikertScorer"):
        await scorer.score_text_async("this has no bullying")
        # RETRY_MAX_NUM_ATTEMPTS is set to 2 in conftest.py
        assert chat_target.send_prompt_async.call_count == 2


# ---------------------------------------------------------------------------
# Custom (non-1-to-5) scale tests
# ---------------------------------------------------------------------------


def _make_custom_scale_yaml(
    tmp_path: Path,
    *,
    category: str = "test_harm",
    min_val: int = 0,
    max_val: int = 7,
) -> Path:
    """
    Create a YAML file with a custom Likert scale range inside a pytest tmp_path.

    The file is automatically cleaned up by pytest after the test session.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        category (str): The category name for the scale.
        min_val (int): Minimum score value.
        max_val (int): Maximum score value.

    Returns:
        Path: Path to the created YAML file.
    """
    scale = {
        "version": "1.0",
        "category": category,
        "scale_descriptions": [
            {"score_value": str(i), "description": f"Level {i} description"} for i in range(min_val, max_val + 1)
        ],
    }
    yaml_file = tmp_path / f"{category}_scale.yaml"
    yaml_file.write_text(yaml.safe_dump(scale), encoding="utf-8")
    return yaml_file


def test_custom_scale_sets_min_max(tmp_path: Path):
    """Verify that a YAML with a 0-7 scale sets _min/_max correctly."""
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        custom_path = _make_custom_scale_yaml(tmp_path, min_val=0, max_val=7)
        # Temporarily patch the LikertScalePaths enum path property
        with patch.object(LikertScalePaths, "path", new_callable=lambda: property(lambda self: Path(custom_path))):
            with patch.object(LikertScalePaths, "evaluation_files", new_callable=lambda: property(lambda self: None)):
                scorer = SelfAskLikertScorer(
                    chat_target=chat_target,
                    likert_scale=LikertScalePaths.CYBER_SCALE,
                )

        assert scorer._min_scale_value == 0
        assert scorer._max_scale_value == 7


def test_default_1_to_5_scale_sets_min_max():
    """Verify that existing 1-5 harm definition YAMLs still set min=1, max=5."""
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        scorer = SelfAskLikertScorer(
            chat_target=chat_target,
            likert_scale=LikertScalePaths.VIOLENCE_SCALE,
        )

        assert scorer._min_scale_value == 1
        assert scorer._max_scale_value == 5


def test_custom_scale_system_prompt_contains_dynamic_range(tmp_path: Path):
    """Verify the system prompt references the custom min/max, not hardcoded 1/5."""
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        custom_path = _make_custom_scale_yaml(tmp_path, min_val=0, max_val=7)
        with patch.object(LikertScalePaths, "path", new_callable=lambda: property(lambda self: Path(custom_path))):
            with patch.object(LikertScalePaths, "evaluation_files", new_callable=lambda: property(lambda self: None)):
                scorer = SelfAskLikertScorer(
                    chat_target=chat_target,
                    likert_scale=LikertScalePaths.CYBER_SCALE,
                )

        # The system prompt should mention the custom range boundaries
        assert "0 is the least severe" in scorer._system_prompt
        assert "7 is the most severe" in scorer._system_prompt


@pytest.mark.asyncio
async def test_custom_scale_score_normalisation(patch_central_database, tmp_path: Path):
    """
    Verify that scoring normalises against the custom range, not 1-5.

    For a 0-7 scale, a raw score of 7 should normalise to 1.0 and
    a raw score of 0 should normalise to 0.0.
    """
    # Simulate LLM returning score_value "7" on a 0-7 scale
    json_response = (
        '{"score_value": "7", "description": "Maximum severity", "rationale": "The response is extremely harmful."}'
    )
    llm_response = Message(message_pieces=[MessagePiece(role="assistant", original_value=json_response)])

    chat_target = MagicMock()
    chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
    chat_target.send_prompt_async = AsyncMock(return_value=[llm_response])

    custom_path = _make_custom_scale_yaml(tmp_path, min_val=0, max_val=7)
    with patch.object(LikertScalePaths, "path", new_callable=lambda: property(lambda self: Path(custom_path))):
        with patch.object(LikertScalePaths, "evaluation_files", new_callable=lambda: property(lambda self: None)):
            scorer = SelfAskLikertScorer(
                chat_target=chat_target,
                likert_scale=LikertScalePaths.CYBER_SCALE,
            )

    score = await scorer.score_text_async("extremely harmful content")

    assert len(score) == 1
    # 7/7 on a 0-7 scale = 1.0
    assert score[0].score_value == "1.0"
    assert score[0].get_value() == 1.0
    assert score[0].score_metadata == {"likert_value": 7}


@pytest.mark.asyncio
async def test_custom_scale_score_min_value(patch_central_database, tmp_path: Path):
    """
    Verify min-range normalisation on a 0-7 scale.

    Raw score 0 on a 0-7 scale should normalise to 0.0.
    """
    json_response = '{"score_value": "0", "description": "No harm", "rationale": "The response is benign."}'
    llm_response = Message(message_pieces=[MessagePiece(role="assistant", original_value=json_response)])

    chat_target = MagicMock()
    chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")
    chat_target.send_prompt_async = AsyncMock(return_value=[llm_response])

    custom_path = _make_custom_scale_yaml(tmp_path, min_val=0, max_val=7)
    with patch.object(LikertScalePaths, "path", new_callable=lambda: property(lambda self: Path(custom_path))):
        with patch.object(LikertScalePaths, "evaluation_files", new_callable=lambda: property(lambda self: None)):
            scorer = SelfAskLikertScorer(
                chat_target=chat_target,
                likert_scale=LikertScalePaths.CYBER_SCALE,
            )

    score = await scorer.score_text_async("benign content")
    assert score[0].score_value == "0.0"
    assert score[0].get_value() == 0.0


def test_likert_scale_negative_value_rejected(tmp_path: Path):
    """Verify that negative score values in a YAML are rejected."""
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        chat_target = MagicMock()
        chat_target.get_identifier.return_value = get_mock_target_identifier("MockChatTarget")

        custom_path = _make_custom_scale_yaml(tmp_path, min_val=-1, max_val=5)
        with patch.object(LikertScalePaths, "path", new_callable=lambda: property(lambda self: Path(custom_path))):
            with patch.object(LikertScalePaths, "evaluation_files", new_callable=lambda: property(lambda self: None)):
                with pytest.raises(ValueError, match="non-negative"):
                    SelfAskLikertScorer(
                        chat_target=chat_target,
                        likert_scale=LikertScalePaths.CYBER_SCALE,
                    )
