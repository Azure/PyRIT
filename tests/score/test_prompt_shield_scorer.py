# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import Mock
from tests.mocks import get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import PromptShieldScorer

@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()

@pytest.fixture
def sample_response_json_str() -> str:
    return "{'userPromptAnalysis':{'attackDetected':false},'documentsAnalysis':[{'attackDetected':true}]}"

@pytest.fixture
def promptshield_scorer() -> PromptShieldScorer:
    return PromptShieldScorer(
        target=Mock()
    )

@pytest.fixture
def sample_delineated_prompt_as_str() -> str:
    sample: str = '''
    Mock userPrompt
    <document>
    mock document
    </document>
    '''
    return sample

@pytest.mark.asyncio
def test_prompt_shield_scorer_parsing(
    promptshield_scorer: PromptShieldScorer,
    sample_response_json_str: str
):
    assert promptshield_scorer._parse_response_to_boolean_list(
        sample_response_json_str
    ) == True
