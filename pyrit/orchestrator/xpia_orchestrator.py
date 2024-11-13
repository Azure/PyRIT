# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from aioconsole import ainput
import concurrent.futures
import logging
from typing import Callable, Optional, Union, Awaitable
from uuid import uuid4

from pyrit.score import Scorer
from pyrit.models import Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


class XPIAOrchestrator(Orchestrator):

    def __init__(
        self,
        *,
        attack_content: str,
        attack_setup_target: PromptTarget,
        processing_callback: Callable[[], Awaitable[str]],
        scorer: Optional[Scorer] = None,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory_labels: dict[str, str] = {},
        verbose: bool = False,
        attack_setup_target_conversation_id: Optional[str] = None,
    ) -> None:
        """Creates an orchestrator to set up a cross-domain prompt injection attack (XPIA) on a processing target.

        The attack_setup_target creates the attack prompt using the attack_content,
        applies converters (if any), and puts it into the attack location.
        Then, the processing_callback is executed.
        The scorer scores the processing response to determine the success of the attack.

        Args:
            attack_content: The content to attack the processing target with, e.g., a jailbreak.
            attack_setup_target: The target that generates the attack prompt and gets it into the attack location.
            processing_callback: The callback to execute after the attack prompt is positioned in the attack location.
                This is generic on purpose to allow for flexibility.
                The callback should return the processing response.
            scorer: The scorer to use to score the processing response. This is optional.
                If no scorer is provided the orchestrator will skip scoring.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            memory_labels (dict[str, str], Optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to an empty dictionary.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(prompt_converters=prompt_converters, memory_labels=memory_labels, verbose=verbose)

        self._attack_setup_target = attack_setup_target
        self._processing_callback = processing_callback

        self._scorer = scorer
        self._prompt_normalizer = PromptNormalizer()
        self._attack_setup_target_conversation_id = attack_setup_target_conversation_id or str(uuid4())
        self._processing_conversation_id = str(uuid4())
        self._attack_content = str(attack_content)

    async def execute_async(self) -> Union[Score, None]:
        """Executes the entire XPIA operation.

        This method sends the attack content to the prompt target, processes the response
        using the processing callback, and scores the processing response using the scorer.
        If no scorer was provided, the method will skip scoring.
        """
        logger.info(
            "Sending the following prompt to the prompt target (after applying prompt "
            f'converter operations) "{self._attack_content}"',
        )

        target_request = self._create_normalizer_request(
            converters=self._prompt_converters, prompt_text=self._attack_content, prompt_type="text"
        )

        response = await self._prompt_normalizer.send_prompt_async(
            normalizer_request=target_request,
            target=self._attack_setup_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        logger.info(f'Received the following response from the prompt target "{response}"')

        processing_response = await self._processing_callback()

        logger.info(f'Received the following response from the processing target "{processing_response}"')

        if not self._scorer:
            logger.info("No scorer provided, skipping scoring")
            return None

        pool = concurrent.futures.ThreadPoolExecutor()
        score = pool.submit(asyncio.run, self._scorer.score_text_async(processing_response)).result()[0]

        logger.info(f"Score of the processing response: {score}")
        return score


class XPIATestOrchestrator(XPIAOrchestrator):
    def __init__(
        self,
        *,
        attack_content: str,
        processing_prompt: str,
        processing_target: PromptTarget,
        attack_setup_target: PromptTarget,
        scorer: Scorer,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory_labels: Optional[dict[str, str]] = None,
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
            attack_content: The content to attack the processing target with, e.g., a jailbreak.
            processing_prompt: The prompt to send to the processing target. This should include
                placeholders to invoke plugins (if any).
            processing_target: The target of the attack which processes the processing prompt.
            attack_setup_target: The target that generates the attack prompt and gets it into the attack location.
            scorer: The scorer to use to score the processing response.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            memory_labels (dict[str, str], Optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to None.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            attack_content=attack_content,
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            processing_callback=self._process_async,  # type: ignore
            prompt_converters=prompt_converters,
            memory_labels=memory_labels,
            verbose=verbose,
            attack_setup_target_conversation_id=attack_setup_target_conversation_id,
        )

        self._processing_target = processing_target
        self._processing_conversation_id = str(uuid4())
        self._processing_prompt = processing_prompt

    async def _process_async(self) -> str:
        processing_prompt_req = self._create_normalizer_request(
            converters=[], prompt_text=self._processing_prompt, prompt_type="text"
        )

        processing_response = await self._prompt_normalizer.send_prompt_async(
            normalizer_request=processing_prompt_req,
            target=self._processing_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        return processing_response.request_pieces[0].converted_value


class XPIAManualProcessingOrchestrator(XPIAOrchestrator):
    def __init__(
        self,
        *,
        attack_content: str,
        attack_setup_target: PromptTarget,
        scorer: Scorer,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory_labels: Optional[dict[str, str]] = {},
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
            attack_content: The content to attack the processing target with, e.g., a jailbreak.
            attack_setup_target: The target that generates the attack prompt and gets it into the attack location.
            scorer: The scorer to use to score the processing response.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            attack_content=attack_content,
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            processing_callback=self._input_async,
            prompt_converters=prompt_converters,
            memory_labels=memory_labels,
            verbose=verbose,
            attack_setup_target_conversation_id=attack_setup_target_conversation_id,
        )

    async def _input_async(self):
        return await ainput("Please trigger the processing target's execution and paste the output here: ")
