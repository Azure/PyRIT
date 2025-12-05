# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import uuid

import pytest

from pyrit.models import MessagePiece
from pyrit.prompt_target import OpenAIResponseTarget


@pytest.mark.asyncio
async def test_openai_responses_gpt5(sqlite_instance):
    args = {
        "endpoint": os.getenv("AZURE_OPENAI_GPT5_RESPONSES_ENDPOINT"),
        "model_name": os.getenv("AZURE_OPENAI_GPT5_MODEL"),
        "api_key": os.getenv("AZURE_OPENAI_GPT5_KEY"),
    }

    target = OpenAIResponseTarget(**args)

    conv_id = str(uuid.uuid4())

    developer_piece = MessagePiece(
        role="developer",
        original_value="You are a helpful assistant.",
        original_value_data_type="text",
        conversation_id=conv_id,
        attack_identifier={"id": str(uuid.uuid4())},
    )
    sqlite_instance.add_message_to_memory(request=developer_piece.to_message())

    user_piece = MessagePiece(
        role="user",
        original_value="What is the capital of France?",
        original_value_data_type="text",
        conversation_id=conv_id,
    )

    result = await target.send_prompt_async(message=user_piece.to_message())
    assert result is not None
    assert len(result) == 1
    assert len(result[0].message_pieces) == 2
    assert result[0].message_pieces[0].role == "assistant"
    assert result[0].message_pieces[1].role == "assistant"
    # Hope that the model manages to give the correct answer somewhere (GPT-5 really should)
    assert "Paris" in result[0].message_pieces[1].converted_value
