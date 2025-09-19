# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import random
import tempfile
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2

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
from pyrit.score.scorer_evaluation.metrics_type import MetricsType

logger = logging.getLogger(__name__)


class Scorer(abc.ABC):
    """
    Abstract base class for scorers.
    """

    scorer_type: ScoreType

    @property
    def _memory(self) -> MemoryInterface:
        return CentralMemory.get_memory_instance()

    def _verify_and_resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Verify that a path passed to a Scorer on its creation
        is valid before beginning the scoring logic.

        Args:
            path (Union[str, Path]): A pathlike argument passed to the Scorer.

        Returns:
            Path: The resolved Path object.
        """
        if not isinstance(path, (str, Path)):
            raise ValueError(f"Path must be a string or Path object. Got type(path): {type(path).__name__}")

        path_obj: Path = Path(path).resolve() if isinstance(path, str) else path.resolve()
        if not path_obj.exists():
            raise ValueError(f"Path not found: {str(path_obj)}")
        return path_obj

    async def score_async(
        self, request_response: PromptRequestPiece, *, task: Optional[str] = None, num_frames: Optional[int] = None
    ) -> list[Score]:
        """
        Score the request_response, add the results to the database
        and return a list of Score objects.

        Args:
            request_response (PromptRequestPiece): The request response to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
            num_frames (int, optional): The number of frames to extract from a video for scoring.
                Only applicable if the request_response is a video.

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        self.validate(request_response, task=task)
        scores: List[Score] = []

        # If the request_response is a video, extract frames and score each frame
        # This handling will no longer be needed once there are models that can score videos in their entirety
        # For now, there only exist models that can score images and text
        if request_response.converted_value_data_type == "video_path":
            scores = await self.score_video_async(
                video_path=request_response.converted_value,
                task=task,
                num_frames=num_frames,
            )

            scores[0].prompt_request_response_id = request_response.id
        else:
            scores = await self._score_async(request_response, task=task)

        self._memory.add_scores_to_memory(scores=scores)
        return scores

    @abstractmethod
    async def _score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
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

    def _extract_frames(self, video_path: str, num_frames: int = 5) -> list[str]:
        """
        Extracts a specified number of image frames from a video file and returns them as a list of (temp)
        image file paths.

        Args:
            video_path (str): The path to the video file.
            num_frames (int): The number of image frames to extract from the video.

        """

        video_capture = cv2.VideoCapture(video_path)
        frame_paths = []
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            # Choose up to num_frames random unique frame indices
            frame_indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))
            for frame_index in frame_indices:
                # Set the video position to the selected frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Failed to read frame at index {frame_index}")
                    continue

                # Create a temporary file for the frame
                with tempfile.NamedTemporaryFile(suffix=f"_frame_{frame_index}.png", delete=False) as temp_file:
                    # Encode and write frame to temporary file
                    ret, _ = cv2.imencode(".png", frame)
                    if not ret:
                        print(f"Failed to encode frame at index {frame_index}")
                        continue

                    cv2.imwrite(temp_file.name, frame)
                    frame_paths.append(temp_file.name)

        video_capture.release()
        return frame_paths

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

    async def score_text_batch_async(
        self,
        *,
        texts: Sequence[str],
        tasks: Optional[Sequence[str]] = None,
        batch_size: int = 10,
    ) -> list[Score]:
        if tasks:
            if len(tasks) != len(texts):
                raise ValueError("The number of tasks must match the number of texts.")
        if len(texts) == 0:
            return []
        prompt_target = getattr(self, "_prompt_target")
        results = await batch_task_async(
            task_func=self.score_text_async,
            task_arguments=["text", "task"] if tasks else ["text"],
            prompt_target=prompt_target,
            batch_size=batch_size,
            items_to_batch=[texts, tasks] if tasks else [texts],
        )
        return [score for sublist in results for score in sublist]

    async def score_image_batch_async(
        self, *, image_paths: Sequence[str], tasks: Optional[Sequence[str]] = None, batch_size: int = 10
    ) -> list[Score]:
        if tasks:
            if len(tasks) != len(image_paths):
                raise ValueError("The number of tasks must match the number of image_paths.")
        if len(image_paths) == 0:
            return []
        prompt_target = getattr(self, "_prompt_target", None)
        results = await batch_task_async(
            task_func=self.score_image_async,
            task_arguments=["image_path", "task"] if tasks else ["image_path"],
            prompt_target=prompt_target,
            batch_size=batch_size,
            items_to_batch=[image_paths, tasks] if tasks else [image_paths],
        )
        return [score for sublist in results for score in sublist]

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
        num_frames: int = 5,
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
            task_arguments=["request_response", "task", "num_frames"],
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

    async def score_video_async(
        self, video_path: str, *, task: Optional[str] = None, num_frames: Optional[int] = None
    ) -> list[Score]:
        """
        Scores the given video by breaking it up into specified number of image frames and passing those through
        to the chat target for evaluation. The frame scores will then be aggregated into one score for the video,
        choosing the most severe score based on the scorer type.

        Args:
            video_path (str): The path to the video to be scored.
            task (str): The task based on which the video should be scored (the original attacker model's objective).

        Returns:
            list[Score]: A list of Score objects representing the results.
        """
        if not num_frames:
            num_frames = 5  # Default to 5 frames if not specified

        # Extract frames from the video and score each frame
        image_frame_paths = self._extract_frames(video_path, num_frames=num_frames)
        if not image_frame_paths:
            raise ValueError("No frames extracted from video for scoring.")

        tasks = [task] * len(image_frame_paths)
        frame_scores = await self.score_image_batch_async(
            image_paths=image_frame_paths, tasks=tasks, batch_size=num_frames
        )
        if not frame_scores:
            raise ValueError("No scores returned for image frames extracted from video.")

        # Clean up temporary frame files
        for path in image_frame_paths:
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Error removing temporary frame file {path}: {e}")

        return self._aggregate_frame_scores(frame_scores)

    def _aggregate_frame_scores(self, frame_scores: list[Score]) -> list[Score]:
        """
        Aggregates a list of frame scores into a single score for the entire video.

        Args:
            frame_scores (list[Score]): The list of frame scores to aggregate.

        Returns:
            Score (list[Score]): The aggregated scores for the entire video.
        """
        # Aggregate frame scores into one score for the entire video
        aggregate_score = None
        original_rationale = None

        # For true_false scorers, if any of the scores are True, the aggregate score is True
        if self.scorer_type == "true_false":
            true_scores = [score for score in frame_scores if score.get_value()]
            true_count = len(true_scores)

            # Update the rationale to reflect the aggregation logic
            if true_count > 0:
                original_rationale = "\nSee original score rationales for each frame as follows:\n* "
                original_rationale += "\n* ".join([score.score_rationale for score in true_scores])

                aggregate_score = true_scores[0]
                aggregate_score.score_rationale = (
                    f"{true_count} out of {len(frame_scores)} video frames meet objective."
                )
            else:
                aggregate_score = frame_scores[0]
                original_rationale = aggregate_score.score_rationale

                aggregate_score.score_rationale = f"Objective not met within {len(frame_scores)} video frames."

        # For float_scale scorers, use the highest score value as the aggregate score value
        elif self.scorer_type == "float_scale":
            aggregate_score = max(frame_scores, key=lambda score: score.get_value())
            original_rationale = aggregate_score.score_rationale

            # Update the rationale to indicate this is the highest score
            highest_score = aggregate_score.get_value()
            aggregate_score.score_rationale = f"Highest score across {len(frame_scores)} frames: {highest_score:.2f}. "

        else:
            raise ValueError(f"Unsupported scorer type for video scoring: {self.scorer_type}")

        if original_rationale:
            # Append the original rationale to the updated rationale
            aggregate_score.score_rationale += original_rationale

        # Set the Score ID
        aggregate_score.id = uuid.uuid4()

        return [aggregate_score]

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
        category: Optional[str] = None,
        task: Optional[str] = None,
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
            task (str, Optional): A description of the task that is associated with the score, used for
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
        skip_on_error: bool = True,
        num_frames: Optional[int] = None,
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
            num_frames: Optional number of frames to extract from a video for scoring. Only applicable
                if the response is a video.

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
        tasks = [
            scorer.score_async(request_response=piece, task=task, num_frames=num_frames)
            for piece in filtered_pieces[:1]
            for scorer in scorers
        ]

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
        num_frames: Optional[int] = None,
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
            num_frames: Optional number of frames to extract from a video for scoring. Only applicable
                if the response is a video.

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
            tasks = [scorer.score_async(request_response=piece, task=task, num_frames=num_frames) for scorer in scorers]
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
        task: Optional[str] = None,
        skip_on_error: bool = True,
        num_frames: Optional[int] = None,
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
            num_frames (Optional[int]): Optional number of frames to extract from a video for scoring.
                Only applicable if the response is a video.

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
                task=task,
                skip_on_error=skip_on_error,
                num_frames=num_frames,
            )

            objective_task = Scorer.score_response_select_first_success_async(
                response=response,
                scorers=objective_scorers,
                role_filter=role_filter,
                task=task,
                skip_on_error=skip_on_error,
                num_frames=num_frames,
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
                task=task,
                skip_on_error=skip_on_error,
                num_frames=num_frames,
            )

        # Run only objective scoring
        elif has_objective:
            objective_score = await Scorer.score_response_select_first_success_async(
                response=response,
                scorers=objective_scorers,
                role_filter=role_filter,
                task=task,
                skip_on_error=skip_on_error,
                num_frames=num_frames,
            )
            result["objective_scores"] = [objective_score] if objective_score else []

        return result
