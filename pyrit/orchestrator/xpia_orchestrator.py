# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Callable, Optional, Union
from uuid import uuid4
from pyrit.score import SupportTextClassification

from pyrit.memory import MemoryInterface
from pyrit.score import Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


class XPIAOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_content: str,
        attack_setup_target: PromptTarget,
        processing_callback: Callable[[], str],
        scorer: Optional[SupportTextClassification] = None,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = None,
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
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._attack_setup_target = attack_setup_target
        self._processing_callback = processing_callback
        self._scorer = scorer

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._attack_setup_target._memory = self._memory
        self._attack_setup_target_conversation_id = attack_setup_target_conversation_id or str(uuid4())
        self._processing_conversation_id = str(uuid4())
        self._attack_content = str(attack_content)

    def execute(self) -> Union[Score, None]:
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

        response = self._prompt_normalizer.send_prompt(
            normalizer_request=target_request,
            target=self._attack_setup_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        logger.info(f'Received the following response from the prompt target "{response}"')

        processing_response = self._processing_callback()

        logger.info(f'Received the following response from the processing target "{processing_response}"')

        if not self._scorer:
            logger.info("No scorer provided, skipping scoring")
            return None
        score = self._scorer.score_text(processing_response)
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
        scorer: SupportTextClassification,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = None,
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
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            attack_content=attack_content,
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            processing_callback=self._process,
            prompt_converters=prompt_converters,
            memory=memory,
            memory_labels=memory_labels,
            verbose=verbose,
            attack_setup_target_conversation_id=attack_setup_target_conversation_id,
        )

        self._processing_target = processing_target
        self._processing_conversation_id = str(uuid4())
        self._processing_prompt = processing_prompt

    def _process(self) -> str:
        processing_prompt_req = self._create_normalizer_request(
            converters=[], prompt_text=self._processing_prompt, prompt_type="text"
        )

        processing_response = self._prompt_normalizer.send_prompt(
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
        scorer: SupportTextClassification,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = None,
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
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
            attack_setup_target_conversation_id: The conversation ID to use for the prompt target.
                If not provided, a new one will be generated.
        """
        super().__init__(
            attack_content=attack_content,
            attack_setup_target=attack_setup_target,
            scorer=scorer,
            processing_callback=self._input,
            prompt_converters=prompt_converters,
            memory=memory,
            memory_labels=memory_labels,
            verbose=verbose,
            attack_setup_target_conversation_id=attack_setup_target_conversation_id,
        )

    def _input(self):
        return input("Please trigger the processing target's execution and paste the output here: ")
