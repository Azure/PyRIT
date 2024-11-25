# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import logging

from typing import Optional
import uuid

from pyrit.exceptions import InvalidJsonException, pyrit_json_retry, remove_markdown_json
from pyrit.memory import MemoryInterface, CentralMemory
from pyrit.models import SeedPrompt
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import NormalizerRequestPiece, PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class TreeOfAttacksNode:
    """
    Creates a Node to be used with Tree of Attacks with Pruning.
    """

    _memory: MemoryInterface

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_seed_prompt: SeedPrompt,
        adversarial_chat_prompt_template: SeedPrompt,
        adversarial_chat_system_seed_prompt: SeedPrompt,
        objective_scorer: Scorer,
        on_topic_scorer: Scorer,
        prompt_converters: list[PromptConverter],
        orchestrator_id: dict[str, str],
        memory_labels: Optional[dict[str, str]] = None,
        parent_id: Optional[str] = None,
    ) -> None:

        self._objective_target = objective_target
        self._adversarial_chat = adversarial_chat
        self._objective_scorer = objective_scorer
        self._adversarial_chat_seed_prompt = adversarial_chat_seed_prompt
        self._adversarial_chat_prompt_template = adversarial_chat_prompt_template
        self._adversarial_chat_system_seed_prompt = adversarial_chat_system_seed_prompt
        self._on_topic_scorer = on_topic_scorer
        self._prompt_converters = prompt_converters
        self._orchestrator_id = orchestrator_id
        self._memory = CentralMemory.get_memory_instance()
        self._global_memory_labels = memory_labels or {}

        self._prompt_normalizer = PromptNormalizer()
        self.parent_id = parent_id
        self.node_id = str(uuid.uuid4())

        self.objective_target_conversation_id = str(uuid.uuid4())
        self.adversarial_chat_conversation_id = str(uuid.uuid4())

        self.prompt_sent = False
        self.completed = False
        self.score = 0.0
        self.off_topic = False

    async def send_prompt_async(self, objective: str):
        """Executes one turn of a branch of a tree of attacks with pruning.

        This includes a few steps. At first, the red teaming target generates a prompt for the prompt target.
        If on-topic checking is enabled, the branch will get pruned if the generated prompt is off-topic.
        If it is on-topic or on-topic checking is not enabled, the prompt is sent to the prompt target.
        The response from the prompt target is finally scored by the scorer.
        """

        self.prompt_sent = True

        try:
            prompt = await self._generate_red_teaming_prompt_async(objective=objective)
        except InvalidJsonException as e:
            logger.error(f"Failed to generate a prompt for the prompt target: {e}")
            logger.info("Pruning the branch since we can't proceed without red teaming prompt.")
            return

        if self._on_topic_scorer:
            on_topic_score = (await self._on_topic_scorer.score_text_async(text=prompt))[0]

            # If the prompt is not on topic we prune the branch.
            if not on_topic_score.get_value():
                self.off_topic = True
                return

        objective_target_request = NormalizerRequest(
            request_pieces=[
                NormalizerRequestPiece(
                    request_converters=self._prompt_converters,
                    prompt_value=prompt,
                    prompt_data_type="text",
                )
            ],
            conversation_id=self.objective_target_conversation_id,
        )

        response = (
            await self._prompt_normalizer.send_prompt_async(
                normalizer_request=objective_target_request,
                target=self._objective_target,
                labels=self._global_memory_labels,
                orchestrator_identifier=self._orchestrator_id,
            )
        ).request_pieces[0]

        logger.debug(f"saving score with prompt_request_response_id: {response.id}")

        self.score = (
            await self._objective_scorer.score_async(
                request_response=response,
                task=objective,
            )
        )[0].get_value()

        self.completed = True

    def duplicate(self) -> TreeOfAttacksNode:
        """
        Creates a duplicate of the provided instance
        with incremented iteration and new conversations ids (but duplicated conversations)
        """
        duplicate_node = TreeOfAttacksNode(
            objective_target=self._objective_target,
            adversarial_chat=self._adversarial_chat,
            adversarial_chat_seed_prompt=self._adversarial_chat_seed_prompt,
            adversarial_chat_prompt_template=self._adversarial_chat_prompt_template,
            adversarial_chat_system_seed_prompt=self._adversarial_chat_system_seed_prompt,
            objective_scorer=self._objective_scorer,
            on_topic_scorer=self._on_topic_scorer,
            prompt_converters=self._prompt_converters,
            orchestrator_id=self._orchestrator_id,
            memory_labels=self._global_memory_labels,
            parent_id=self.node_id,
        )

        duplicate_node.objective_target_conversation_id = self._memory.duplicate_conversation(
            conversation_id=self.objective_target_conversation_id
        )

        duplicate_node.adversarial_chat_conversation_id = self._memory.duplicate_conversation(
            conversation_id=self.adversarial_chat_conversation_id,
        )

        return duplicate_node

    @pyrit_json_retry
    async def _generate_red_teaming_prompt_async(self, objective) -> str:

        # Use the red teaming target to generate a prompt for the attack target.
        # The prompt for the red teaming target needs to include the latest message from the prompt target.
        # A special case is the very first message, in which case there are no prior messages
        # so we can use the initial red teaming prompt
        target_messages = self._memory.get_conversation(conversation_id=self.objective_target_conversation_id)

        if not target_messages:
            self._adversarial_chat.set_system_prompt(
                system_prompt=self._adversarial_chat_system_seed_prompt.render_template_value(objective=objective),
                conversation_id=self.adversarial_chat_conversation_id,
                orchestrator_identifier=self._orchestrator_id,
                labels=self._global_memory_labels,
            )

            logger.debug("Using the specified initial red teaming prompt for the first turn.")
            prompt_text = self._adversarial_chat_seed_prompt.render_template_value(objective=objective)

        else:
            assistant_responses = [r for r in target_messages if r.request_pieces[0].role == "assistant"]
            if not assistant_responses:
                logger.error(
                    f"No assistant responses found in the conversation {self.objective_target_conversation_id}."
                )
                raise RuntimeError("Cannot proceed without an assistant response.")

            target_response = assistant_responses[-1]
            target_response_piece = target_response.request_pieces[0]
            logger.debug(f"target_response_piece.id: {target_response_piece.id}")
            scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(target_response_piece.id)])

            if scores:
                score = scores[0].get_value()
            else:
                score = "unavailable"
            prompt_text = self._adversarial_chat_prompt_template.render_template_value(
                target_response=target_response_piece.converted_value,
                objective=objective,
                score=str(score),
            )

        adversarial_chat_request = NormalizerRequest(
            request_pieces=[
                NormalizerRequestPiece(request_converters=[], prompt_value=prompt_text, prompt_data_type="text")
            ],
            conversation_id=self.adversarial_chat_conversation_id,
        )

        adversarial_chat_response = (
            (
                await self._prompt_normalizer.send_prompt_async(
                    normalizer_request=adversarial_chat_request,
                    target=self._adversarial_chat,
                    labels=self._global_memory_labels,
                    orchestrator_identifier=self._orchestrator_id,
                )
            )
            .request_pieces[0]
            .converted_value
        )

        return self._parse_red_teaming_response(adversarial_chat_response)

    def _parse_red_teaming_response(self, red_teaming_response: str) -> str:
        # The red teaming response should be in JSON format with two keys: "prompt" and "improvement".
        # We need to parse only "prompt" and return its value.

        # If the JSON is valid in Markdown format, remove the Markdown formatting
        red_teaming_response = remove_markdown_json(red_teaming_response)
        try:
            red_teaming_response_dict = json.loads(red_teaming_response)
        except json.JSONDecodeError:
            logger.error(f"The response from the red teaming chat is not in JSON format: {red_teaming_response}")
            raise InvalidJsonException(message="The response from the red teaming chat is not in JSON format.")

        try:
            return red_teaming_response_dict["prompt"]
        except KeyError:
            logger.error(f"The response from the red teaming chat does not contain a prompt: {red_teaming_response}")
            raise InvalidJsonException(message="The response from the red teaming chat does not contain a prompt.")

    def __str__(self) -> str:
        return (
            "TreeOfAttackNode("
            f"completed={self.completed}, "
            f"score={self.score}, "
            f"node_id={self.node_id}, "
            f"objective_target_conversation_id={self.objective_target_conversation_id})"
        )

    __repr__ = __str__
