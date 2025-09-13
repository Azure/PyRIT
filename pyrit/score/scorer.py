# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import asyncio
import json
import logging
import uuid
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Union, cast

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
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async
from pyrit.score.scorer_evaluation.metrics_type import MetricsType
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator

logger = logging.getLogger(__name__)


class Scorer(abc.ABC):
    """
    Abstract base class for scorers.
    """

    scorer_type: ScoreType

    def __init__(self, *, validator: ScorerPromptValidator):
        self._validator = validator

    @property
    def _memory(self) -> MemoryInterface:
        return CentralMemory.get_memory_instance()

    async def score_async(
        self,
        request_response: PromptRequestResponse,
        *,
        objective: Optional[str] = None,
        role_filter: Optional[ChatMessageRole] = None,
    ) -> list[Score]:
        """
        Score the request_response, add the results to the database
        and return a list of Score objects.

        Args:
            request_response (PromptRequestResponse): The request response to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        self._validator.validate(request_response, objective=objective)
        scores = await self._score_async(
            request_response,
            objective=objective,
            role_filter=role_filter,
        )
        self._memory.add_scores_to_memory(scores=scores)

        self.validate_return_scores(scores=scores)

        return scores

    @abstractmethod
    async def _score_async(
        self,
        request_response: PromptRequestResponse,
        *,
        objective: Optional[str] = None,
        role_filter: Optional[ChatMessageRole] = None,
    ) -> list[Score]:
        raise NotImplementedError()
    
    @abstractmethod
    async def _score_piece_async(self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None) -> list[Score]:
        raise NotImplementedError()
    

    def validate_request(self, request_piece: PromptRequestResponse, *, objective: Optional[str] = None) -> None:
        """Validate the request/response prior to scoring.

        Ensures there is at least one request piece available. Child scorers will
        perform their own specific validations during their scoring.

        Args:
            request_piece (PromptRequestResponse): The request/response to validate.
            objective (Optional[str]): Unused here; may be used by child scorers.

        Raises:
            ValueError: If the request/response is missing pieces.
        """
        if request_piece is None or not getattr(request_piece, "request_pieces", None):
            raise ValueError("PromptRequestResponse must contain at least one request piece.")



    def _get_supported_pieces(self, request_response: PromptRequestResponse) -> list[PromptRequestPiece]:
        """
        Returns a list of supported request pieces for this scorer.
        """
        return [piece for piece in request_response.request_pieces 
                if self._validator.is_request_piece_supported(request_piece=piece)]

    
    @abstractmethod
    def validate_return_scores(self, scores: list[Score]):
        """
        Validates the scores returned by the scorer. Because some scorers may require
        specific Score types or values.

        Args:
            scores (list[Score]): The scores to be validated.
        """
        raise NotImplementedError()

    def get_scorer_metrics(self, dataset_name: str, metrics_type: Optional[MetricsType] = None):
        """
        Returns evaluation statistics for the scorer using the dataset_name of the human labeled dataset that this
        scorer was run against. If you did not evaluate the scorer against your own human labeled dataset, you can
        use this method to retrieve metrics based on a pre-existing dataset name, which is often a 'harm_category'
        or abbreviated version of the 'objective'. For example, to retrieve metrics for the 'hate_speech' harm,
        you would pass 'hate_speech' as the dataset_name.

        The existing metrics can be found in the 'dataset/score/scorer_evals' directory within either
        the 'harm' or 'objective' subdirectory.

        Args:
            dataset_name (str): The name of the dataset on which the scorer evaluation was run. This is used to
                inform the name of the metrics file to read in the `scorer_evals` directory.
            metrics_type (MetricsType, optional): The type of metrics to retrieve, either HARM
                or OBJECTIVE. If not provided, it will default to OBJECTIVE for true/false scorers
                and HARM for all other scorers.

        Returns:
            ScorerMetrics: A ScorerMetrics object containing the saved evaluation statistics for the scorer.
        """
        # Import ScorerEvaluator here to avoid circular imports
        from pyrit.score import ScorerEvaluator

        scorer_evaluator = ScorerEvaluator.from_scorer(self, metrics_type=metrics_type)
        return scorer_evaluator.get_scorer_metrics(dataset_name=dataset_name)

    async def score_text_async(self, text: str, *, objective: Optional[str] = None) -> list[Score]:
        """
        Scores the given text based on the task using the chat target.

        Args:
            text (str): The text to be scored.
            objective (str): The task based on which the text should be scored

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value=text,
                )
            ]
        )

        request.request_pieces[0].id = None
        return await self.score_async(request, objective=objective)
    
    async def score_image_async(self, image_path: str, *, objective: Optional[str] = None) -> list[Score]:
        """
        Scores the given image using the chat target.

        Args:
            text (str): The image to be scored.
            objective (str): The objective based on which the text should be scored

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value=image_path,
                    original_value_data_type="image_path",
                )
            ]
        )

        request.request_pieces[0].id = None
        return await self.score_async(request, objective=objective)


    async def score_prompts_batch_async(
        self,
        *,
        request_responses: Sequence[PromptRequestResponse],
        tasks: Sequence[str],
        batch_size: int = 10,
    ) -> list[Score]:
        if not tasks:
            raise ValueError("Objectives must be provided.")
        if len(tasks) != len(request_responses):
            raise ValueError("The number of tasks must match the number of request_responses.")

        if len(request_responses) == 0:
            return []

        # Some scorers do not have an associated prompt target; batch helper validates RPM only when present
        prompt_target = getattr(self, "_prompt_target", None)
        results = await batch_task_async(
            task_func=self.score_async,
            task_arguments=["request_response", "objective"],
            prompt_target=cast(PromptTarget, prompt_target),
            batch_size=batch_size,
            items_to_batch=[request_responses, tasks],
        )

        # results is a list[list[Score]] and needs to be flattened
        return [score for sublist in results for score in sublist]



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
    async def _score_value_with_llm(
        self,
        *,
        prompt_target: PromptChatTarget,
        system_prompt: str,
        prompt_request_value: str,
        prompt_request_data_type: PromptDataType,
        scored_prompt_id: str,
        category: Optional[Sequence[str] | str] = None,
        objective: Optional[str] = None,
        score_value_output_key: str = "score_value",
        rationale_output_key: str = "rationale",
        description_output_key: str = "description",
        metadata_output_key: str = "metadata",
        category_output_key: str = "category",
        orchestrator_identifier: Optional[Dict[str, str]] = None,
    ) -> UnvalidatedScore:
        """
        Sends a request to a target, and takes care of retries.

        The scorer target response should be JSON with value, rationale, and optional metadata and
        description fields.

        Args:
            prompt_target (PromptChatTarget): The target LLM to send the prompt request to.
            system_prompt (str): The system-level prompt that guides the behavior of the target LLM.
            prompt_request_value (str): The actual value or content to be scored by the LLM.
            prompt_request_data_type (PromptDataType): The type of the data being sent in the prompt request.
            scored_prompt_id (str): The ID of the scored prompt.
            category (str, Optional): The category of the score. Can also be parsed from the JSON response if
                not provided.
            objective (str, Optional): A description of the objective that is associated with the score, used for
                contextualizing the result.
            score_value_output_key (str): The key in the JSON response that contains the score value.
            rationale_output_key (str): The key in the JSON response that contains the rationale.
            description_output_key (str): The key in the JSON response that contains the description.
            category_output_key (str): The key in the JSON response that contains the category.
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

        response_json: str = ""
        try:
            response_json = response.get_value()

            response_json = remove_markdown_json(response_json)
            parsed_response = json.loads(response_json)
            category_response = parsed_response.get(category_output_key)

            if category_response and category:
                raise ValueError("Category is present in the response and an argument")

            # Validate and normalize category to a list of strings
            cat_val = category_response if category_response is not None else category
            normalized_category: Optional[list[str]]
            if cat_val is None:
                normalized_category = None
            elif isinstance(cat_val, str):
                normalized_category = [cat_val]
            elif isinstance(cat_val, list):
                if not all(isinstance(x, str) for x in cat_val):
                    raise ValueError("'category' must be a string or a list of strings")
                normalized_category = cat_val
            else:
                # JSON must yield either a string or a list of strings
                raise ValueError("'category' must be a string or a list of strings")

            # Normalize metadata to a dictionary with string keys and string/int values
            raw_md = parsed_response.get(metadata_output_key)
            normalized_md: Optional[Dict[str, Union[str, int]]]
            if raw_md is None:
                normalized_md = None
            elif isinstance(raw_md, dict):
                # Coerce keys to str and filter to str/int values only
                normalized_md = {
                    str(k): v
                    for k, v in raw_md.items()
                    if isinstance(v, (str, int))
                }
                # If dictionary becomes empty after filtering, keep as empty dict
            elif isinstance(raw_md, (str, int)):
                # Wrap primitive metadata into a namespaced field
                normalized_md = {"metadata": raw_md}
            else:
                # Unrecognized metadata shape; drop to avoid downstream errors
                normalized_md = None

            score = UnvalidatedScore(
                raw_score_value=str(parsed_response[score_value_output_key]),
                score_value_description=parsed_response.get(description_output_key),
                score_category=normalized_category,
                score_rationale=parsed_response[rationale_output_key],
                scorer_class_identifier=self.get_identifier(),
                score_metadata=normalized_md,
                prompt_request_response_id=scored_prompt_id,
                objective=objective,
            )

        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON response: {response_json}")

        except KeyError:
            raise InvalidJsonException(message=f"Invalid JSON response, missing Key: {response_json}")

        return score


    ###################
    # It probably makes sense to remove all below this
    ###################

    @staticmethod
    async def score_response_async(
        *,
        response: PromptRequestResponse,
        scorers: List[Scorer],
        role_filter: ChatMessageRole = "assistant",
        task: Optional[str] = None,
        skip_on_error: bool = True,
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
            skip_on_error: If True, skip scoring pieces that have errors (default: True)

        Returns:
            List of all scores from all scorers
        """
        if not scorers:
            return []

        # Filter response pieces by role
        filtered_pieces = list(response.filter_by_role(role=role_filter))
        if not filtered_pieces:
            return []

        # Further filter out error responses if requested
        if skip_on_error:
            filtered_pieces = [p for p in filtered_pieces if not p.has_error()]
            if not filtered_pieces:
                logger.debug("All response pieces have errors, skipping scoring")
                return []

        # Create all scoring tasks, note TEMPORARY fix to prevent multi-piece responses from breaking scoring logic
        tasks = []
        for piece in filtered_pieces[:1]:
            single = PromptRequestResponse([piece])
            for scorer in scorers:
                tasks.append(scorer.score_async(request_response=single, objective=task))

        if not tasks:
            return []

        # Execute all tasks in parallel
        score_lists = await asyncio.gather(*tasks)

        # Flatten the list of lists into a single list
        return [score for scores in score_lists for score in scores]

    @staticmethod
    async def score_response_select_first_success_async(
        *,
        response: PromptRequestResponse,
        scorers: List[Scorer],
        role_filter: ChatMessageRole = "assistant",
        task: Optional[str] = None,
        skip_on_error: bool = True,
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
            skip_on_error: If True, skip scoring pieces that have errors (default: True)

        Returns:
            The first successful score, or the first score if no success found, or None if no scores
        """
        if not scorers:
            return None

        # Filter response pieces by role
        filtered_pieces = list(response.filter_by_role(role=role_filter))
        if not filtered_pieces:
            return None

        # Further filter out error responses if requested
        if skip_on_error:
            scorable_pieces = [p for p in filtered_pieces if not p.has_error()]
            if not scorable_pieces:
                logger.debug("All response pieces have errors, skipping scoring")
                return None
        else:
            scorable_pieces = filtered_pieces

        first_score = None

        # TEMPORARY fix to prevent multi-piece responses from breaking scoring logic of attack
        for piece in scorable_pieces[:1]:
            # Run all scorers on this piece in parallel
            single = PromptRequestResponse([piece])
            tasks = [scorer.score_async(request_response=single, objective=task) for scorer in scorers]
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

    @staticmethod
    async def score_response_with_objective_async(
        *,
        response: PromptRequestResponse,
        auxiliary_scorers: Optional[List[Scorer]] = None,
        objective_scorers: Optional[List[Scorer]] = None,
        role_filter: ChatMessageRole = "assistant",
        objective: Optional[str] = None,
        skip_on_error: bool = True,
    ) -> Dict[str, List[Score]]:
        """
        Score a response using both auxiliary and objective scorers.

        This method runs auxiliary scorers for collecting metrics and objective scorers
        for determining success. All scorers are run asynchronously for performance.

        Args:
            response (PromptRequestResponse): Response containing pieces to score
            auxiliary_scorers (Optional[List[Scorer]]): List of auxiliary scorers to apply
            objective_scorers (Optional[List[Scorer]]): List of objective scorers to apply
            role_filter (ChatMessageRole): Only score pieces with this role (default: `assistant`)
            task (Optional[str]): Optional task description for scoring context
            skip_on_error (bool): If True, skip scoring pieces that have errors (default: `True`)

        Returns:
            Dict[str,List[Score]]: Dictionary with keys `auxiliary_scores` and `objective_scores`
                containing lists of scores from each type of scorer.
        """
        # Initialize result dictionary
        result: Dict[str, List[Score]] = {"auxiliary_scores": [], "objective_scores": []}

        has_auxiliary = auxiliary_scorers is not None
        has_objective = objective_scorers is not None

        # Early return if no scorers provided
        if not has_auxiliary and not has_objective:
            return result

        # Run both types of scoring concurrently if both are present
        if has_auxiliary and has_objective:
            auxiliary_task = Scorer.score_response_async(
                response=response,
                scorers=auxiliary_scorers,
                role_filter=role_filter,
                task=objective,
                skip_on_error=skip_on_error,
            )

            objective_task = Scorer.score_response_select_first_success_async(
                response=response,
                scorers=objective_scorers,
                role_filter=role_filter,
                task=objective,
                skip_on_error=skip_on_error,
            )

            # Run them in parallel and unpack results
            auxiliary_scores, objective_score = await asyncio.gather(auxiliary_task, objective_task)

            # Store results
            result["auxiliary_scores"] = auxiliary_scores
            result["objective_scores"] = [objective_score] if objective_score else []

        # Run only auxiliary scoring
        elif has_auxiliary:
            result["auxiliary_scores"] = await Scorer.score_response_async(
                response=response,
                scorers=auxiliary_scorers,
                role_filter=role_filter,
                task=objective,
                skip_on_error=skip_on_error,
            )

        # Run only objective scoring
        elif has_objective:
            objective_score = await Scorer.score_response_select_first_success_async(
                response=response,
                scorers=objective_scorers,
                role_filter=role_filter,
                task=objective,
                skip_on_error=skip_on_error,
            )
            result["objective_scores"] = [objective_score] if objective_score else []

        return result
