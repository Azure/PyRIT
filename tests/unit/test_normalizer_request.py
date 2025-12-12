# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import uuid

import pytest

from pyrit.models import Message
from pyrit.prompt_normalizer import NormalizerRequest


def test_normalizer_request_validates_sequence():
    # Test that NormalizerRequest accepts messages with consistent sequences
    message = Message.from_prompt(prompt="Hello", role="user")
    request = NormalizerRequest(
        message=message,
        conversation_id=str(uuid.uuid4()),
    )
    
    # Verify request was created successfully
    assert request.message == message
    assert len(request.message.message_pieces) == 1


def test_normalizer_request_validates_empty_message():
    # Test that Message constructor itself validates non-empty message_pieces
    with pytest.raises(ValueError) as exc_info:
        message = Message(message_pieces=[])

    assert "Message must have at least one message piece" in str(exc_info.value)
