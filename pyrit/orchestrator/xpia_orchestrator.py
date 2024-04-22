# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional
from uuid import uuid4
from pyrit.interfaces import SupportTextClassification

from pyrit.memory import MemoryInterface
from pyrit.models import Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


class XPIATestOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_content: str,
        processing_prompt: str,
        processing_target: PromptTarget,
        prompt_target: PromptTarget,
        scorer: SupportTextClassification,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = None,
        verbose: bool = False,
        prompt_target_conversation_id: Optional[str] = None,
    ) -> None:
        """Creates an orchestrator to set up a cross-domain prompt injection attack (XPIA) on a processing target.

        The prompt_target creates the attack prompt using the attack_content,
        applies converters (if any), and puts it into the attack location.
        The processing_target processes the processing_prompt which should include
        a reference to the attack prompt to allow it to retrieve the attack prompt.
        The scorer scores the processing response to determine the success of the attack.

        Args:
            attack_content: The content to attack the processing target with, e.g., a jailbreak.
            processing_prompt: The prompt to send to the processing target. This should include
                placeholders to invoke plugins (if any).
            processing_target: The target of the attack which processes the processing prompt.
            prompt_target: The target that generates the attack prompt and gets it into the attack location.
            scorer: The scorer to use to score the processing response.
            prompt_converters: The converters to apply to the attack content before sending it to the prompt target.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """
        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target
        self._processing_target = processing_target
        self._scorer = scorer

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = prompt_target_conversation_id or str(uuid4())
        self._processing_conversation_id = str(uuid4())
        self._attack_content = str(attack_content)
        self._processing_prompt = processing_prompt

    def process(self) -> Score:
        logger.info(
            "Sending the following prompt to the prompt target (after applying prompt "
            f'converter operations) "{self._attack_content}"',
        )

        target_request = self._create_normalizer_request(
            converters=self._prompt_converters, prompt_text=self._attack_content, prompt_type="text"
        )

        response = self._prompt_normalizer.send_prompt(
            normalizer_request=target_request,
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        logger.info(f'Received the following response from the prompt target "{response}"')

        processing_prompt_req = self._create_normalizer_request(
            converters=[], prompt_text=self._processing_prompt, prompt_type="text"
        )

        processing_response = self._prompt_normalizer.send_prompt(
            normalizer_request=processing_prompt_req,
            target=self._processing_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        logger.info(f'Received the following response from the processing target "{processing_response}"')

        score = self._scorer.score_text(processing_response.request_pieces[0].converted_prompt_text)
        logger.info(f"Score of the processing response: {score}")
        return score
