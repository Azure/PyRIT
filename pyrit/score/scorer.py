# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import asyncio
import json
import uuid
from abc import abstractmethod
from typing import List, Optional, Sequence

from pyrit.exceptions import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import (
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    ScoreType,
    UnvalidatedScore,
)
from pyrit.models.literals import ChatMessageRole
from pyrit.prompt_target import PromptChatTarget
from pyrit.prompt_target.batch_helper import batch_task_async


class Scorer(abc.ABC):
    """
    Abstract base class for scorers.
    """

    scorer_type: ScoreType

    @property
    def _memory(self) -> MemoryInterface:
        return CentralMemory.get_memory_instance()

    @abstractmethod
    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Score the request_response, add the results to the database
        and return a list of Score objects.

        Args:
            request_response (PromptRequestPiece): The request response to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        raise NotImplementedError("score_async method not implemented")

    @abstractmethod
    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        """
        Validates the request_response piece to score. Because some scorers may require
        specific PromptRequestPiece types or values.

        Args:
            request_response (PromptRequestPiece): The request response to be validated.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
        """
        raise NotImplementedError("score_async method not implemented")

    async def score_text_async(self, text: str, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given text based on the task using the chat target.

        Args:
            text (str): The text to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        request_piece = PromptRequestPiece(
            role="user",
            original_value=text,
        )

        request_piece.id = None
        return await self.score_async(request_piece, task=task)

    async def score_responses_inferring_tasks_batch_async(
        self,
        *,
        request_responses: Sequence[PromptRequestPiece],
        batch_size: int = 10,
    ) -> list[Score]:
        """
        Scores a batch of responses (ignores non-assistant messages).

        This will send the last requests as tasks if it can. If it's complicated (e.g. non-text) it will send None.

        For more control, use score_prompts_with_tasks_batch_async
        """
        responses = [piece for piece in request_responses if piece.role == "assistant"]
        tasks = [self._extract_task_from_response(response) for response in responses]
        return await self.score_prompts_with_tasks_batch_async(
            request_responses=responses, tasks=tasks, batch_size=batch_size
        )

    async def score_prompts_with_tasks_batch_async(
        self,
        *,
        request_responses: Sequence[PromptRequestPiece],
        tasks: Sequence[str],
        batch_size: int = 10,
    ) -> list[Score]:
        if not tasks:
            raise ValueError("Tasks must be provided.")
        if len(tasks) != len(request_responses):
            raise ValueError("The number of tasks must match the number of request_responses.")

        if len(request_responses) == 0:
            return []

        prompt_target = getattr(self, "_prompt_target", None)
        results = await batch_task_async(
            task_func=self.score_async,
            task_arguments=["request_response", "task"],
            prompt_target=prompt_target,
            batch_size=batch_size,
            items_to_batch=[request_responses, tasks],
        )

        # results is a list[list[Score]] and needs to be flattened
        return [score for sublist in results for score in sublist]

    async def score_image_async(self, image_path: str, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given image using the chat target.

        Args:
            text (str): The image to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        request_piece = PromptRequestPiece(
            role="user",
            original_value=image_path,
            converted_value=image_path,
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
        )

        request_piece.id = None
        return await self.score_async(request_piece, task=task)

    def scale_value_float(self, value: float, min_value: float, max_value: float) -> float:
        """
        Scales a value from 0 to 1 based on the given min and max values. E.g. 3 stars out of 5 stars would be .5.

        Args:
            value (float): The value to be scaled.
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.

        Returns:
            float: The scaled value.
        """
        if max_value == min_value:
            return 0.0

        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_identifier(self):
        """
        Returns an identifier dictionary for the scorer.

        Returns:
            dict: The identifier dictionary.
        """
        identifier = {}
        identifier["__type__"] = self.__class__.__name__
        identifier["__module__"] = self.__class__.__module__
        identifier["sub_identifier"] = None
        return identifier

    def _extract_task_from_response(self, response: PromptRequestPiece) -> str:
        """
        Extracts a task from the response using the last request (if it exists).

        Args:
            response (PromptRequestPiece): The response to extract the task from.

        Returns:
            str: The task extracted from the response.
        """
        if response.role != "assistant":
            return ""

        conversation = self._memory.get_prompt_request_pieces(conversation_id=response.conversation_id)

        # Every text request piece from the last turn
        last_turn_text = "\n".join(
            [
                piece.original_value
                for piece in conversation
                if piece.sequence == response.sequence - 1 and piece.original_value_data_type == "text"
            ]
        )

        return last_turn_text

    @pyrit_json_retry
    async def _score_value_with_llm(
        self,
        *,
        prompt_target: PromptChatTarget,
        system_prompt: str,
        prompt_request_value: str,
        prompt_request_data_type: PromptDataType,
        scored_prompt_id: str,
        category: str = None,
        task: str = None,
        score_value_output_key: Optional[str] = "score_value",
        rationale_output_key: Optional[str] = "rationale",
        description_output_key: Optional[str] = "description",
        metadata_output_key: Optional[str] = "metadata",
        category_output_key: Optional[str] = "category",
        orchestrator_identifier: dict[str, str] = None,
    ) -> UnvalidatedScore:
        """
        Sends a request to a target, and takes care of retries.

        The scorer target response should be JSON with value, rationale, and optional metadata and description fields.

        Args:
            prompt_target (PromptChatTarget): The target LLM to send the prompt request to.
            system_prompt (str): The system-level prompt that guides the behavior of the target LLM.
            prompt_request_value (str): The actual value or content to be scored by the LLM.
            prompt_request_data_type (PromptDataType): The type of the data being sent in the prompt request.
            scored_prompt_id (str): The ID of the scored prompt.
            category (str, Optional): The category of the score. Can also be parsed from the JSON response if not
                provided.
            task (str, Optional): A description of the task that is associated with the score, used for contextualizing
                the result.
            score_value_output_key (str, Optional): The key in the JSON response that contains the score value.
            rationale_output_key (str, Optional): The key in the JSON response that contains the rationale.
            description_output_key (str, Optional): The key in the JSON response that contains the description.
            metadata_output_key (str, Optional): The key in the JSON response that contains the metadata.
            orchestrator_identifier (dict[str, str], Optional): A dictionary containing orchestrator-specific
                identifiers.

        Returns:
            UnvalidatedScore: The score object containing the response from the target LLM.
                score_value still needs to be normalized and validated.
        """

        conversation_id = str(uuid.uuid4())

        if orchestrator_identifier:
            orchestrator_identifier["scored_prompt_id"] = str(scored_prompt_id)

        prompt_target.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=orchestrator_identifier,
        )
        prompt_metadata: dict[str, str | int] = {"response_format": "json"}
        scorer_llm_request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=prompt_request_value,
                    original_value_data_type=prompt_request_data_type,
                    converted_value_data_type=prompt_request_data_type,
                    conversation_id=conversation_id,
                    prompt_target_identifier=prompt_target.get_identifier(),
                    prompt_metadata=prompt_metadata,
                )
            ]
        )
        try:
            response = await prompt_target.send_prompt_async(prompt_request=scorer_llm_request)
        except Exception as ex:
            raise Exception(f"Error scoring prompt with original prompt ID: {scored_prompt_id}") from ex

        try:
            response_json = response.get_value()

            response_json = remove_markdown_json(response_json)
            parsed_response = json.loads(response_json)
            category_response = parsed_response.get(category_output_key)

            if category_response and category:
                raise ValueError("Category is present in the response and an argument")

            category = category_response if category_response else category

            score = UnvalidatedScore(
                raw_score_value=str(parsed_response[score_value_output_key]),
                score_value_description=parsed_response.get(description_output_key),
                score_type=self.scorer_type,
                score_category=category,
                score_rationale=parsed_response[rationale_output_key],
                scorer_class_identifier=self.get_identifier(),
                score_metadata=parsed_response.get(metadata_output_key),
                prompt_request_response_id=scored_prompt_id,
                task=task,
            )

        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON response: {response_json}")

        except KeyError:
            raise InvalidJsonException(message=f"Invalid JSON response, missing Key: {response_json}")

        try:
            if self.scorer_type == "float_scale":
                # raise an exception if it's not parsable as a float
                float(score.raw_score_value)
        except ValueError:
            raise InvalidJsonException(
                message=f"Invalid JSON response, score_value should be a float not this: {score.raw_score_value}"
            )

        return score

    @staticmethod
    async def score_response_async(
        *,
        response: PromptRequestResponse,
        scorers: List[Scorer],
        role_filter: ChatMessageRole = "assistant",
        task: Optional[str] = None,
    ) -> List[Score]:
        """
        Score a response using multiple scorers in parallel.

        This method runs all scorers on all filtered response pieces concurrently for maximum performance.
        Typically used for auxiliary scoring where all results are needed but not returned.

        Args:
            response: PromptRequestResponse containing pieces to score
            scorers: List of scorers to apply
            role_filter: Only score pieces with this role (default: "assistant")
            task: Optional task description for scoring context

        Returns:
            List of all scores from all scorers
        """
        if not scorers:
            return []

        # Filter response pieces by role
        filtered_pieces = list(response.filter_by_role(role=role_filter))
        if not filtered_pieces:
            return []

        # Create all scoring tasks
        tasks = [
            scorer.score_async(request_response=piece, task=task) for piece in filtered_pieces for scorer in scorers
        ]

        if not tasks:
            return []

        # Execute all tasks in parallel
        score_lists = await asyncio.gather(*tasks)

        # Flatten the list of lists into a single list
        return [score for scores in score_lists for score in scores]

    @staticmethod
    async def score_response_until_success_async(
        *,
        response: PromptRequestResponse,
        scorers: List[Scorer],
        role_filter: ChatMessageRole = "assistant",
        task: Optional[str] = None,
    ) -> Optional[Score]:
        """
        Score response pieces sequentially until finding a successful score.

        This method processes filtered response pieces one by one. For each piece, it runs all
        scorers in parallel, then checks the results for a successful score (where score.get_value()
        is truthy). If no successful score is found, it returns the first score as a failure indicator.

        Args:
            response: PromptRequestResponse containing pieces to score
            scorers: List of scorers to use for evaluation
            role_filter: Only score pieces with this role (default: "assistant")
            task: Optional task description for scoring context

        Returns:
            The first successful score, or the first score if no success found, or None if no scores
        """
        if not scorers:
            return None

        # Filter response pieces by role
        filtered_pieces = list(response.filter_by_role(role=role_filter))
        if not filtered_pieces:
            return None

        first_score = None

        for piece in filtered_pieces:
            # Run all scorers on this piece in parallel
            tasks = [scorer.score_async(request_response=piece, task=task) for scorer in scorers]
            score_lists = await asyncio.gather(*tasks)

            # Flatten the results
            scores = [score for scores in score_lists for score in scores]

            # Remember the first score as potential fallback
            if scores and first_score is None:
                first_score = scores[0]

            # Check for successful score
            for score in scores:
                if score.get_value():
                    return score

        # No successful score found - return first score as failure indicator
        return first_score
