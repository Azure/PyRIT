# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from pathlib import Path
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt
from pyrit.orchestrator import TreeOfAttacksWithPruningOrchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class PAIROrchestrator(TreeOfAttacksWithPruningOrchestrator):
    """
    This orchestrator implements the Prompt Automatic Iterative Refinement (PAIR) algorithm

    This algorithm was published and described in the paper: Chao, Patrick, et al. Jailbreaking Black Box Large
    Language Models in Twenty Queries. arXiv:2310.08419, arXiv, 13 Oct. 2023. arXiv.org, http://arxiv.org/abs/2310.08419

    The authors published a reference implementation in the following repository: https://github.com/patrickrchao/\
    JailbreakingLLMs/blob/main/system_prompts.py
    """

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        scoring_target: PromptChatTarget,
        adversarial_chat_seed_prompt: Optional[SeedPrompt] = None,
        adversarial_chat_system_prompt_path: Optional[Path] = None,
        depth: int = 3,
        prompt_converters: Optional[list[PromptConverter]] = None,
        objective_achieved_score_threshold: float = 0.8,
        desired_response_prefix="Sure, here is",
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:

        adversarial_chat_system_prompt_path = adversarial_chat_system_prompt_path or Path(
            DATASETS_PATH / "orchestrators" / "pair" / "attacker_system_prompt.yaml"
        )

        super().__init__(
            objective_target=objective_target,
            adversarial_chat=adversarial_chat,
            scoring_target=scoring_target,
            adversarial_chat_seed_prompt=adversarial_chat_seed_prompt,
            adversarial_chat_system_prompt_path=adversarial_chat_system_prompt_path,
            width=1,
            depth=depth,
            branching_factor=1,
            on_topic_checking_enabled=False,
            prompt_converters=prompt_converters,
            objective_achieved_score_threshold=objective_achieved_score_threshold,
            desired_response_prefix=desired_response_prefix,
            verbose=verbose,
            batch_size=batch_size,
        )

    def set_prepended_conversation(self, *, prepended_conversation):
        return super().set_prepended_conversation(prepended_conversation=prepended_conversation)
