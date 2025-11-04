# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pydantic import BaseModel

from pyrit.models import (
    Message,
    MessagePiece,
)
from pyrit.prompt_target import (
    OpenAIResponseTarget,
    PromptChatTarget,
    limit_requests_per_minute,
)
from pyrit.score import AzureContentFilterScorer

logger = logging.getLogger(__name__)


class Beam(BaseModel):
    id: str
    text: str
    score: float


class BeamSearchScorer:
    def __init__(self, scorer: AzureContentFilterScorer, beam_keep_frac: float):
        self.scorer = scorer
        assert 0.0 < beam_keep_frac <= 1.0, "beam_keep_frac must be in (0.0, 1.0]"
        self.beam_keep_frac = beam_keep_frac

    async def _score_beam(self, beam: Beam):
        scores = await self.scorer.score_text_async(beam.text)
        total_score = 0.0
        for s in scores:
            total_score += s.get_value()
        beam.score = total_score

    async def _cull_and_replace(self, beams: list[Beam]):
        # Cull the lowest scoring beam and replace it with a new one
        beams.sort(key=lambda b: b.score)

        beams[0].id = beams[-1].id
        keep_chars = int(len(beams[-1].text) * self.beam_keep_frac)
        beams[0].text = beams[-1].text[:keep_chars]
        return beams

    async def update_beams(self, beams: list[Beam]) -> list[Beam]:
        for beam in beams:
            await self._score_beam(beam)
        updated_beams = await self._cull_and_replace(beams)
        return updated_beams


class BeamSearchTarget(PromptChatTarget):
    def __init__(
        self, base_target: OpenAIResponseTarget, num_beams: int, num_steps: int, num_chars_per_step: int
    ) -> None:
        """
        A beam search target that wraps an OpenAIResponseTarget to provide beam search capabilities.

        Args:
            base_target (OpenAIResponseTarget): The base OpenAI response target to wrap.
        """
        super().__init__(
            endpoint=base_target._endpoint,
            model_name=base_target._model_name,
            max_requests_per_minute=base_target._max_requests_per_minute,
        )
        self._base_target = base_target
        self._num_beams = num_beams
        self._num_steps = num_steps
        self._num_chars_per_step = num_chars_per_step

    def is_json_response_supported(self) -> bool:
        """Let's not have JSON complicate things yet"""
        return False

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: Message) -> Message:
        # Implement beam search logic here
        pass
