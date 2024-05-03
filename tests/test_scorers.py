# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest

from pyrit.score import LikertScales
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score.self_ask_likert_scorer import SelfAskLikertScorer


@pytest.fixture
def chat_target() -> AzureOpenAIChatTarget:
    target = MagicMock()
    return target


def test_likert_scorer_set_system_prompt(chat_target: AzureOpenAIChatTarget):
    with patch.object(AzureOpenAIChatTarget, "set_system_prompt") as mock_method:

        scorer = SelfAskLikertScorer(
            chat_target=chat_target,
            likert_scale_path=LikertScales.CYBER_SCALE.value,
        )

        # assert that the set_system_prompt method was called
        mock_method.assert_called_once()

        # assert that the likert score was loaded into system prompt
        assert scorer._score_category in scorer._system_prompt
        assert "No cyber harm" in scorer._system_prompt

def test_likert_scorer_score():
    target = MagicMock()

    scorer = SelfAskLikertScorer(
        chat_target=chat_target,
        likert_scale_path=LikertScales.CYBER_SCALE.value,
    )

    # assert that the set_system_prompt method was called
    mock_method.assert_called_once()

    # assert that the likert score was loaded into system prompt
    assert scorer._score_category in scorer._system_prompt
    assert "No cyber harm" in scorer._system_prompt
