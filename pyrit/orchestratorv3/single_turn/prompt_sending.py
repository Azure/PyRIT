# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import List, Optional

from pyrit.common.logger import logger
from pyrit.common.utils import combine_dict
from pyrit.models import PromptDataType, PromptRequestResponse
from pyrit.orchestratorv3.base.attack_strategy import AttackStrategy
from pyrit.orchestratorv3.base.core import (
    SingleTurnAttackContext,
    SingleTurnAttackResult,
)
from pyrit.orchestratorv3.components.conversation_manager import ConversationManager
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer


class PromptSendingAttackStrategy(AttackStrategy[SingleTurnAttackContext, SingleTurnAttackResult]):
    """
    Strategy for sending prompts to a target

    This strategy handles preparing, sending, and optionally scoring prompt responses
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        prompt_converters: Optional[List[PromptConverter]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        scorers: Optional[List[Scorer]] = None,
    ):
        """
        Initialize the strategy

        Args:
            prompt_normalizer: Optional custom prompt normalizer
            prepended_conversation: Optional conversation to prepend
            memory: Optional memory interface
            scorers: Optional list of scorers
        """
        super().__init__(logger=logger)
        self._objective_target = objective_target
        self._prompt_converters = prompt_converters or []
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._scorers = scorers or []
        self._conversation_manager = ConversationManager(
            orchestrator_identifier=self.get_identifier(),
        )

    async def _setup(self, *, context: SingleTurnAttackContext) -> None:
        """
        Prepare the strategy for the single-turn attack. This method is called before execution
        to make sure memory labels are up to date and any additional setup tasks are performed

        Args:
            context (SingleTurnAttackContext): The context for the single-turn attack,
            containing target information, memory labels, batch size, and other relevant data.
        """
        # Get updated memory labels
        context.memory_labels = combine_dict(existing_dict=self._memory_labels, new_dict=context.memory_labels or {})

    async def _perform_attack(self, *, context: SingleTurnAttackContext) -> SingleTurnAttackResult:
        """
        Executes a single-turn attack by generating prompt requests, sending them to the target,
        and optionally scoring the responses

        Args:
            context (SingleTurnAttackContext): The context for the single-turn attack,
            containing target information, memory labels, batch size, and other relevant data.

        Returns:
            SingleTurnAttackResult: The result of the single-turn attack, including the list of
            responses received from the target.
        """

        # Generate prompt requests based on the context
        # 1) Build normalized prompt requests
        # 2) Send requests to target in batches
        # 3) Score and return the response results
        requests = await self._get_prompt_requests(
            context=context,
        )

        # Send prompts
        responses = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=requests,
            target=self._objective_target,
            labels=context.memory_labels,
            orchestrator_identifier=self.get_identifier(),
            batch_size=context.batch_size,
        )

        # Score responses if scorers were provided
        await self._score_responses(responses=responses, context=context)

        # Create and return the result
        return SingleTurnAttackResult(orchestrator_identifier=self.get_identifier(), prompt_list=responses)

    async def _teardown(self, *, context: SingleTurnAttackContext) -> None:
        """
        Clean up after the attack execution. This method is called after the attack to ensure
        any resources are released and the state is reset

        Args:
            context (SingleTurnAttackContext): The context for the single-turn attack,
            containing target information, memory labels, batch size, and other relevant data.
        """
        # Nothing to do here for now
        pass

    async def _get_prompt_requests(
        self,
        *,
        context: SingleTurnAttackContext,
        prompt_type: PromptDataType = "text",
    ) -> List[NormalizerRequest]:
        """
        Create prompt requests for the given context.
        Args:
            context: The context for the attack
            prompt_type: The type of prompt to create
        Returns:
            List of NormalizerRequest objects
        """
        if not context.prompts:
            raise ValueError("No prompts configured for attack")

        requests = PromptNormalizer.build_normalizer_requests(
            prompts=context.prompts,
            prompt_type=prompt_type,
            converters=self._prompt_converters,
            metadata=context.metadata,
        )

        # Handle prepended conversation if needed
        if context.prepended_conversation:
            for request in requests:
                await self._prepare_conversation(context=context, normalizer_request=request)

        return requests

    async def _score_responses(
        self, *, responses: List[PromptRequestResponse], context: SingleTurnAttackContext
    ) -> None:
        """
        Score the responses using the provided scorers.

        Args:
            responses: The responses to score
            context: The context for the attack
        """
        if not self._scorers or not responses:
            return

        response_pieces = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)

        for scorer in self._scorers:
            await scorer.score_responses_inferring_tasks_batch_async(
                request_responses=response_pieces, batch_size=context.batch_size
            )

    async def _prepare_conversation(
        self, *, context: SingleTurnAttackContext, normalizer_request: NormalizerRequest
    ) -> None:
        """
        Prepares the conversation by initializing it with the prepended conversation.

        Args:
            normalizer_request: The normalizer request to prepare
        """
        if not context.prepended_conversation:
            return

        # Initialize conversation with prepended conversation
        conversation_id = normalizer_request.conversation_id
        self._conversation_manager.initialize_conversation_with_history(
            conversation_id=conversation_id,
            history=context.prepended_conversation,
        )
