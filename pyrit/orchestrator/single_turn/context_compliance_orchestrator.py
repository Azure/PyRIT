# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
import re
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestPiece, PromptRequestResponse, SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import PromptConverter, SearchReplaceConverter
from pyrit.prompt_normalizer import NormalizerRequest
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ContextDescriptionPaths(enum.Enum):
    GENERAL = pathlib.Path(DATASETS_PATH) / "orchestrators" / "context_compliance" / "context_description.yaml"


class ContextComplianceOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements a context compliance attack
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        context_description_instructions_path: Optional[pathlib.Path] = None,
        scorers: Optional[list[Scorer]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target for sending prompts.
            adversarial_chat (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are only applied
                to the conversation history since in this orchestrator, the latest prompt is usually just "yes"
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        self._adversarial_chat = adversarial_chat

        affirmitive_response = [
            "yes.",
            "Yes",
            "YES",
            "Yes",
            "Y",
            "y",
            "Yup",
            "yup",
            "Yup.",
            "yup.",
            "Yup!",
            "yup!",
            "Sure",
            "sure",
            "Sure.",
            "sure.",
            "Of course",
            "of course",
            "Of course.",
            "of course.",
        ]

        search_replace_converter = SearchReplaceConverter(
            pattern=r"^.*\Z", replace=affirmitive_response, regex_flags=re.DOTALL
        )

        if context_description_instructions_path is None:
            context_description_instructions_path = ContextDescriptionPaths.GENERAL.value

        context_description_instructions: SeedPromptDataset = SeedPromptDataset.from_yaml_file(
            context_description_instructions_path
        )
        self._rephrase_objective_to_user_turn = context_description_instructions.prompts[0]
        self._answer_user_turn = context_description_instructions.prompts[1]
        self._rephrase_objective_to_question = context_description_instructions.prompts[2]

        # these are applied to all adversarial chat responses, and ultimately to the objective target requests
        self._conversation_history_converters = (
            [PromptConverterConfiguration(converters=prompt_converters)] if prompt_converters else []
        )

        super().__init__(
            objective_target=objective_target,
            prompt_converters=[search_replace_converter],
            scorers=scorers,
            batch_size=batch_size,
            verbose=verbose,
        )

    async def get_prepended_conversation_async(
        self, normalizer_request: NormalizerRequest
    ) -> Optional[list[PromptRequestResponse]]:
        """
        Returns the user turn prompts for the given list of prompts.

        This works better if broken into three prompts. One to rephrase the objective as more benign. One to answer
        the more benign objective. And one to rephrase the objective as a question. Ultimately these are assembled into
        a fake context that we send to the objective target.

        Args:
            prompt_list (list[str]): The list of prompts to be role played.

        Returns:
            list[str]: The role playing prompts.
        """

        objective = normalizer_request.seed_prompt_group.prompts[0].value

        benign_user_query = await self._get_objective_as_more_benign_question(objective=objective)
        benign_user_query_answer = await self._get_benign_question_answer(benign_user_query=benign_user_query)
        objective_as_question = await self._get_objective_as_question(objective=objective)

        assistant_response = f"{benign_user_query_answer}  {objective_as_question}"

        return [
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value=objective,
                        converted_value=benign_user_query,
                    ),
                ],
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value=assistant_response,
                    ),
                ],
            ),
        ]

    def validate_normalizer_requests(self, *, prompt_request_list: list[NormalizerRequest]):
        if not prompt_request_list:
            raise ValueError("No normalizer requests provided")

        for request in prompt_request_list:
            if len(request.seed_prompt_group.prompts) > 1:
                raise ValueError("Multi-part messages not supported")
            if request.seed_prompt_group.prompts[0].data_type != "text":
                raise ValueError("Non text messages not supported")

    async def _get_benign_question_answer(self, benign_user_query: str) -> str:
        seed_prompt_to_get_user_turn_answer = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=self._answer_user_turn.render_template_value(benign_request=benign_user_query),
                    data_type="text",
                )
            ]
        )

        user_turn_answer = (
            (
                await self._prompt_normalizer.send_prompt_async(
                    seed_prompt_group=seed_prompt_to_get_user_turn_answer,
                    target=self._adversarial_chat,
                    response_converter_configurations=self._conversation_history_converters,
                )
            )
            .request_pieces[0]
            .converted_value
        )

        return user_turn_answer

    async def _get_objective_as_more_benign_question(self, objective: str) -> str:
        seed_prompt_to_get_user_turn = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=self._rephrase_objective_to_user_turn.render_template_value(objective=objective),
                    data_type="text",
                )
            ]
        )

        user_turn = (
            (
                await self._prompt_normalizer.send_prompt_async(
                    seed_prompt_group=seed_prompt_to_get_user_turn,
                    target=self._adversarial_chat,
                    response_converter_configurations=self._conversation_history_converters,
                )
            )
            .request_pieces[0]
            .converted_value
        )

        return user_turn

    async def _get_objective_as_question(self, objective: str) -> str:
        seed_prompt_to_get_objective_as_a_question = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=self._rephrase_objective_to_question.render_template_value(objective=objective),
                    data_type="text",
                )
            ]
        )

        objective_as_question = (
            (
                await self._prompt_normalizer.send_prompt_async(
                    seed_prompt_group=seed_prompt_to_get_objective_as_a_question,
                    target=self._adversarial_chat,
                    response_converter_configurations=self._conversation_history_converters,
                )
            )
            .request_pieces[0]
            .converted_value
        )

        return objective_as_question
