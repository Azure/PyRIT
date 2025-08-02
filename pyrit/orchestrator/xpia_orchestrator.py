# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import concurrent.futures
import logging
from typing import Awaitable, Callable, Optional, Union
from uuid import uuid4

from aioconsole import ainput

from pyrit.memory import CentralMemory
from pyrit.models import SeedPromptGroup, PromptRequestResponse, PromptRequestPiece
from pyrit.orchestrator import Orchestrator, OrchestratorResult
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class XPIAOrchestrator(Orchestrator):

    def __init__(
        self,
        *,
        attack_setup_target: PromptTarget,
        processing_callback: Callable[[], Awaitable[str]],
        scorer: Optional[Scorer] = None,
        prompt_converters: Optional[list[PromptConverter]] = None,
        verbose: bool = False,
        attack_setup_target_conversation_id: Optional[str] = None,
    ) -> None:
        """Creates an orchestrator to set up a cross-domain prompt injection attack (XPIA) on a processing target.

        The attack_setup_target creates the attack prompt using the attack_content,
        applies converters (if any), and puts it into the attack location.
        Then, the processing_callback is executed.
        The scorer scores the processing response to determine the success of the attack.

        Args:
            attack_setup_target: The target that generates the attack prompt and gets it into the attack location.
            processing_callback: The callback to execute after the attack prompt is positioned in the attack location.
                This is generic on purpose to allow for flexibility.
                The callback should return the processing response.
            scorer: The scorer to use to score the processing response. This is optional.
                If no scorer is provided the orchestrator will skip scoring.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(prompt_converters=prompt_converters, verbose=verbose)

        self._attack_setup_target = attack_setup_target
        self._processing_callback = processing_callback

        self._scorer = scorer
        self._prompt_normalizer = PromptNormalizer()
        self._attack_setup_target_conversation_id = attack_setup_target_conversation_id or str(uuid4())
        self._processing_conversation_id = str(uuid4())

    async def run_attack_async(
        self,
        *,
        seed_prompt: SeedPromptGroup,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:
        """
        Runs the entire XPIA attack from setup to processing and scoring.

        This method sends the attack content to the prompt target, processes the response
        using the processing callback, and scores the processing response using the scorer.
        If no scorer was provided, the method will skip scoring.

        Args:
            seed_prompt (SeedPromptGroup): The seed prompt group to start the conversation.
            memory_labels (dict[str, str], Optional): The memory labels to use for the attack.
        """
        self._validate_seed_prompt(seed_prompt)
        piece = seed_prompt.prompts[0]
    
        logger.info(
            "Sending the following prompt to the prompt target (after applying prompt "
            f'converter operations) "{piece.value}"',
        )

        converters = PromptConverterConfiguration(converters=self._prompt_converters)

        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt,
            request_converter_configurations=[converters],
            target=self._attack_setup_target,
            labels=memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        logger.info(f'Received the following response from the prompt target "{response}"')

        processing_response = await self._processing_callback()
        # manually adding this response to the memory since it's not handled by the normalizer
        memory = CentralMemory.get_memory_instance()
        memory.add_request_response_to_memory(
            request=PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        conversation_id=self._processing_conversation_id,
                        original_value=processing_response,
                        original_value_data_type="text",
                        role="assistant",
                        orchestrator_identifier=self.get_identifier(),
                    )
                ],
            )
        )

        logger.info(f'Received the following response from the processing target "{processing_response}"')

        if not self._scorer:
            logger.info("No scorer provided.")
            return OrchestratorResult(
                conversation_id=self._processing_conversation_id,
                objective=piece.value,
                status="unknown"
            )

        pool = concurrent.futures.ThreadPoolExecutor()
        score = pool.submit(asyncio.run, self._scorer.score_text_async(processing_response)).result()[0]

        logger.info(f"Score of the processing response: {score}")
        return OrchestratorResult(
            conversation_id=self._processing_conversation_id,
            objective=piece.value,
            status="success" if score.get_value() else "failure",
            objective_score=score,
        )

    def _validate_seed_prompt(self, seed_prompt: SeedPromptGroup) -> None:
        if not seed_prompt.prompts or len(seed_prompt.prompts) != 1:
            raise ValueError("Exactly one seed prompt must be provided.")
        prompt = seed_prompt.prompts[0]
        if prompt.data_type != "text":
            raise ValueError(f"Seed prompt must be of type 'text'. Received: {prompt.data_type}")


class XPIATestOrchestrator(XPIAOrchestrator):
    def __init__(
        self,
        *,
        processing_prompt: SeedPromptGroup,
        processing_target: PromptTarget,
        attack_setup_target: PromptTarget,
        scorer: Scorer,
        prompt_converters: Optional[list[PromptConverter]] = None,
        verbose: bool = False,
        attack_setup_target_conversation_id: Optional[str] = None,
    ) -> None:
        """Creates an orchestrator to set up a cross-domain prompt injection attack (XPIA) on a processing target.

        The attack_setup_target creates the attack prompt using the attack_content,
        applies converters (if any), and puts it into the attack location.
        The processing_target processes the processing_prompt which should include
        a reference to the attack prompt to allow it to retrieve the attack prompt.
        The scorer scores the processing response to determine the success of the attack.

        Args:
            processing_prompt: The prompt to send to the processing target. This should include
                prompt request pieces to invoke tools such as input_file or similar.
            processing_target: The target of the attack which processes the processing prompt.
            attack_setup_target: The target that generates the attack prompt and gets it into the attack location.
            scorer: The scorer to use to score the processing response.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            processing_callback=self._process_async,  # type: ignore
            prompt_converters=prompt_converters,
            verbose=verbose,
            attack_setup_target_conversation_id=attack_setup_target_conversation_id,
        )

        self._processing_target = processing_target
        self._processing_conversation_id = str(uuid4())
        self._processing_prompt = processing_prompt

    async def _process_async(self) -> str:
        processing_response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=self._processing_prompt,
            target=self._processing_target,
            orchestrator_identifier=self.get_identifier(),
        )

        return processing_response.get_value()


class XPIAManualProcessingOrchestrator(XPIAOrchestrator):
    def __init__(
        self,
        *,
        attack_setup_target: PromptTarget,
        scorer: Scorer,
        prompt_converters: Optional[list[PromptConverter]] = None,
        verbose: bool = False,
        attack_setup_target_conversation_id: Optional[str] = None,
    ) -> None:
        """Creates an orchestrator to set up a cross-domain prompt injection attack (XPIA) on a processing target.

        The attack_setup_target creates the attack prompt using the attack_content,
        applies converters (if any), and puts it into the attack location.
        Then, the orchestrator stops to wait for the operator to trigger the processing target's execution.
        The operator should paste the output of the processing target into the console.
        Finally, the scorer scores the processing response to determine the success of the attack.

        Args:
            attack_setup_target: The target that generates the attack prompt and gets it into the attack location.
            scorer: The scorer to use to score the processing response.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            processing_callback=self._input_async,
            prompt_converters=prompt_converters,
            verbose=verbose,
            attack_setup_target_conversation_id=attack_setup_target_conversation_id,
        )

    async def _input_async(self):
        return await ainput("Please trigger the processing target's execution and paste the output here: ")
