# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from abc import abstractmethod
import json
from typing import Optional, Sequence

from pyrit.common.batch_helper import batch_task_async
from pyrit.exceptions.exception_classes import InvalidJsonException, pyrit_json_retry
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget
from pyrit.score import Score, ScoreType


class Scorer(abc.ABC):
    """
    Abstract base class for scorers.
    """

    scorer_type: ScoreType

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

    async def score_prompts_batch_async(
        self, prompts: Sequence[PromptRequestPiece], batch_size: int = 10
    ) -> list[Score]:
        prompt_target = getattr(self, "_prompt_target", None)
        results = await batch_task_async(
            task=self.score_async,
            task_argument="request_response",
            prompt_target=prompt_target,
            batch_size=batch_size,
            items_to_batch=prompts,
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

    @pyrit_json_retry
    async def send_chat_target_async(
        self,
        *,
        prompt_target: PromptChatTarget,
        scorer_llm_response: PromptRequestResponse,
        scored_prompt_id: str,
        category: str,
        task: str = "",
    ) -> Score:
        """
        Sends a request to a target LLM, and takes care of retries.

        The scorer LLM response should be JSON with value, rationale, and optional metadata and description fields.

        Args:
            prompt_target (PromptChatTarget): The target LLM to send the prompt request to.
            scorer_llm_response (PromptRequestPiece): The prompt request to be sent to the target LLM.
            scored_prompt_id (str): The ID of the scored prompt.
            category (str): The category of the score.
        Returns:
            Score: The score object containing the response from the target LLM.
        """
        response = await prompt_target.send_prompt_async(prompt_request=scorer_llm_response)

        try:
            response_json = response.request_pieces[0].converted_value
            parsed_response = json.loads(response_json)

            score = Score(
                score_value=str(parsed_response["value"]),
                score_value_description=parsed_response.get("description"),
                score_type=self.scorer_type,
                score_category=category,
                score_rationale=parsed_response["rationale"],
                scorer_class_identifier=self.get_identifier(),
                score_metadata=parsed_response.get("metadata"),
                prompt_request_response_id=scored_prompt_id,
                task=task,
            )
        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON response: {response_json}")

        except KeyError:
            raise InvalidJsonException(message=f"Invalid JSON response, missing Key: {response_json}")

        return score
