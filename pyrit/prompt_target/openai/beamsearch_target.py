# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from pydantic import BaseModel

from pyrit.exceptions import (
    PyritException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
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
        self, base_target: OpenAIResponseTarget, num_beams: int, num_steps: int, num_chars_per_step: int, beam_scorer: BeamSearchScorer
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
        self._search_scorer = beam_scorer
        self._truncate_error_frac = 0.8

    def is_json_response_supported(self) -> bool:
        """Let's not have JSON complicate things yet"""
        return False

    async def send_prompt_async(self, *, prompt_request: Message) -> Message:
        # Implement beam search logic here
        beams = [
            Beam(id=prompt_request.conversation_id, text="", score=0.0) for _ in range(self._num_beams)
        ]

        try:
            for i in range(self._num_steps):
                print(f"Iteration: {i}")
                async with asyncio.TaskGroup() as tg:
                    tasks = [tg.create_task(self._update_beam_text(beam, prompt_request)) for beam in beams]
                    await asyncio.gather(*tasks)

                beams = await self._search_scorer.update_beams(beams)
        finally:
            print("Final beams:")
            for beam in beams:
                print(f"Beam ID: {beam.id}\nText: {beam.text}\nScore: {beam.score}")

        best_beam = beams[-1]
        result = self._memory.get_conversation(conversation_id=best_beam.id)[-1]
        print(f"{result=}")
        return result
    
    async def _update_beam_text(self, beam: Beam, prompt_request: Message):
        new_conversation_id = self._memory.duplicate_conversation(beam.id)

        grammar_template = """
start: PREFIX CONTINUATION
PREFIX: "{prefix}"
CONTINUATION: /.{{0,{n_chars}}}/
"""

        lark_grammar = grammar_template.format(
            prefix=beam.text.replace('"', '\\"'), n_chars=self._num_chars_per_step
        )

        
        grammar_tool = {
            "type": "custom",
            "name": "ContinuationGrammar",
            "description": "Forces continuation of the given prefix.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": lark_grammar,
            },
        }

        reasoning = {"effort": "minimal"}

        ebp = {
            "reasoning": reasoning,
            "tools": [grammar_tool],
            "tool_choice": "required",
        }

        target = self._base_target.fresh_instance()
        target._extra_body_parameters = ebp
        target._grammar_name = grammar_tool["name"]

        try:
            result = await target.send_prompt_async(prompt_request=prompt_request)
            assert len(result.message_pieces) == 2
            beam.text = result.message_pieces[1].original_value
            beam.id = new_conversation_id
        except Exception as e:
            logger.error(f"Error updating beam {beam.id}: {e}")
            print(f"==\n{lark_grammar}\n==")
            new_length = int(len(beam.text) * self._truncate_error_frac)
            beam.text = beam.text[:new_length]