# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import AsyncMock, MagicMock

from pyrit.models import PromptRequestPiece
from pyrit.score import Scorer


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_per_minute", [None, 10])
@pytest.mark.parametrize("batch_size", [1, 10])
async def test_score_prompts_batch_async(requests_per_minute: int, batch_size: int):
    chat_target = MagicMock()
    chat_target._requests_per_minute = requests_per_minute

    scorer = MagicMock(Scorer)
    scorer.score_async = AsyncMock(return_value=["test score"])
    prompt = PromptRequestPiece(role="assistant", original_value="test")

    if batch_size != 1 and requests_per_minute:
        with pytest.raises(ValueError):
            await scorer.score_prompts_batch_async(prompts=[prompt], batch_size=batch_size)
    else:
        results = await scorer.score_prompts_batch_async(prompts=[prompt], batch_size=batch_size)
        assert len(results) == 1
    
    # TODO: Not working with mocks, may need to move this into specific concrete test