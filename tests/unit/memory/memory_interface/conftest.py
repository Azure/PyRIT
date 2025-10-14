# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unit.mocks import get_sample_conversation_entries, get_sample_conversations

from pyrit.models import PromptRequestResponse


@pytest.fixture
def sample_conversations():
    conversations = get_sample_conversations()
    return PromptRequestResponse.flatten_to_prompt_request_pieces(conversations)


@pytest.fixture
def sample_conversation_entries():
    return get_sample_conversation_entries()
