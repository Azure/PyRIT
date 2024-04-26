# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.score import SelfAskGptClassifier, ContentClassifiers
from pyrit.prompt_target import AzureOpenAIChatTarget


@pytest.fixture
def chat_target() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


def test_scorer_set_system_prompt(chat_target: AzureOpenAIChatTarget):
    with patch.object(AzureOpenAIChatTarget, "set_system_prompt") as mock_method:

        scorer = SelfAskGptClassifier(
            content_classifier=ContentClassifiers.PROMPT_INJECTION_CLASSIFIER, chat_target=chat_target
        )

        # assert that the set_system_prompt method was called
        mock_method.assert_called_once()

        # assert that the PROMPT_INJECTION_CLASSIFIER were loaded into system prompt
        assert "regular_text" in scorer._system_prompt
        assert "prompt_injection" in scorer._system_prompt
