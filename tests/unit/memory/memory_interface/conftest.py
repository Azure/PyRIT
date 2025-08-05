import pytest
from unit.mocks import get_sample_conversation_entries, get_sample_conversations

@pytest.fixture
def sample_conversations():
    return get_sample_conversations()

@pytest.fixture
def sample_conversation_entries():
    return get_sample_conversation_entries()