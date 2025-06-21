from pathlib import Path
from typing import Optional

import enum
import json
import yaml
import uuid

from pyrit.common.path import SCALES_PATH
from pyrit.models import (
    PromptDataType,
    PromptRequestPiece, 
    Score, 
    SeedPrompt, 
    UnvalidatedScore, 
    PromptRequestResponse,
)
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer
from pyrit.exceptions import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)

class PolicyScorer(Scorer):
    """
    A scorer that generates evaluation policies and scores model outputs based on them, with human-in-the-loop adjustment.
    """

    class SystemPaths(enum.Enum):
        POLICY_PROMPT = Path(SCALES_PATH, "policy_generator_prompt.yaml").resolve()
        SCORE_PROMPT = Path(SCALES_PATH, "policy_system_prompt.yaml").resolve()

    def __init__(self,
                 *,
                 policy_generator: PromptChatTarget,
                 evaluator: PromptChatTarget,
                 scale_args,
                 policy_generation_prompt_path: Optional[Path] = None,
                 score_prompt_path: Optional[Path] = None,
                 ) -> None:
        self._policy_generator = policy_generator
        self._evaluator = evaluator
        self.scorer_type = "float_scale"

        if not policy_generation_prompt_path:
            policy_generation_prompt_path = self.SystemPaths.POLICY_PROMPT.value
        if not score_prompt_path:
            score_prompt_path = self.SystemPaths.SCORE_PROMPT.value

        self._validate_scale_arguments_set(scale_args)
        self._minimum_value = scale_args["minimum_value"]
        self._maximum_value = scale_args["maximum_value"]
        self._category = scale_args["category"]
        self._task = scale_args["task"]

        generating_policies_template = SeedPrompt.from_yaml_file(policy_generation_prompt_path)
        self.scoring_instructions_template = SeedPrompt.from_yaml_file(score_prompt_path)

        self._policy_prompt = generating_policies_template.render_template_value_silent(**scale_args)
        self.policies = None

    async def _generate_policies(self):
        """
        Generates evaluation policies using the policy generation LLM.
        """
        policy_request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=self._policy_prompt,
                )
            ]
        )
        
        try:
            response = await self._policy_generator.send_prompt_async(prompt_request=policy_request)
        except Exception as ex:
            raise Exception(f"Error policy prompt") from ex
        
        try:
            response_json = response.request_pieces[0].converted_value

            response_json = remove_markdown_json(response_json)
            parsed_response = json.loads(response_json)

            policies = parsed_response['policy']
        except Exception as ex:
            raise InvalidJsonException(f"Invalid JSON response: {response_json}")
        
        return policies
    
    async def score_async(self, request_response, *, task = None):
        if not self.policies:
            self.policies = await self._generate_policies()
            self.weights = self._adjust_weight_manually()

        # render scoring template based on generated criteria
        scoring_args = {
            "minimum_value": self._minimum_value,
            "maximum_value": self._maximum_value,
            "policies": '- ' + '\n- '.join(self.policies),
        }

        system_prompt = self.scoring_instructions_template.render_template_value(**scoring_args)
        scoring_prompt = f"task: {task}\nresponse: {request_response.converted_value}"

        unvalidated_scores = await self._score_dimension_value_with_llm(
            prompt_target=self._evaluator,
            system_prompt=system_prompt,
            prompt_request_value=scoring_prompt,
            prompt_request_data_type=request_response.converted_value_data_type,
            scored_prompt_id=request_response.id,
            category=self._category,
            task=task,
        )
        # Validate score values
        validated_scores = []
        for s in unvalidated_scores:
            score = s.to_score(
                score_value=str(
                    self.scale_value_float(
                        float(s.raw_score_value), self._minimum_value, self._maximum_value
                    )
                )
            )
            validated_scores.append(score)
            
        # weighted final score
        final_score_value = 0
        for i, s in enumerate(validated_scores):
            final_score_value += self.weights[i] * s.get_value()
        final_score_value /= sum(self.weights)
        final_score = Score(
            score_value=str(final_score_value),
            score_value_description='- '+ '\n- '.join([s.score_value_description for s in unvalidated_scores]),
            score_type=self.scorer_type,
            score_category=self._category,
            score_rationale='- '+ '\n- '.join([s.score_rationale for s in unvalidated_scores]),
            score_metadata="Policies:\n- " + '\n- '.join(self.policies),
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id)
        self._memory.add_scores_to_memory(scores=[final_score])
        return [final_score]


    @pyrit_json_retry
    async def _score_dimension_value_with_llm(
        self,
        *,
        prompt_target: PromptChatTarget,
        system_prompt: str,
        prompt_request_value: str,
        prompt_request_data_type: PromptDataType,
        scored_prompt_id: str,
        category: str = None,
        task: str = None,
        orchestrator_identifier: dict[str, str] = None,
    ) -> list[UnvalidatedScore]:
        """
        Sends a request to LLM for multi-policy scoring and returns a list of UnvalidatedScores in the same order as policy_dimensions.
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

            scores = parsed_response["score"]
            descriptions = parsed_response["descriptions"]            
            rationales = parsed_response["rationales"]

            if not (len(scores) == len(self.policies)):
                raise ValueError(f"Mismatch between number of scores and policy dimensions. Got {len(scores)} scores but expected {len(policy_dimensions)}.")

            results = []
            for i, policy in enumerate(self.policies):
                results.append(
                    UnvalidatedScore(
                        raw_score_value=str(scores[i]),
                        score_value_description=descriptions[i] if i < len(descriptions) else None,
                        score_type="float_scale",
                        score_category=self._category,
                        score_rationale=rationales[i] if i < len(rationales) else None,
                        scorer_class_identifier=self.get_identifier(),
                        score_metadata=None,
                        prompt_request_response_id=scored_prompt_id,
                        task=task,
                    )
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise InvalidJsonException(message=f"Invalid or malformed JSON response: {response_json}") from e
        

        return results


    def _adjust_weight_manually(self) -> list[float]:
        '''
        Manually adjust weight of each policy
        '''
        weights = []
        for policy in self.policies:
            weight = ""
            while not weight:
                message = f"The task is: {self._task}\nThe policy is: {policy}\nPlease enter a weight value between '0.0' and '1.0:"
                weight = self._get_user_input(message)
                try:
                    value = self._validate_human_weight(weight)
                    weights.append(value)
                except ValueError as e:
                    print(e)
                    weight = ""
        return weights  


    def _get_user_input(self, message) -> str:
        try:
            import tkinter as tk
            from tkinter import simpledialog
        except ImportError as e:
            print(
                "To adjust weight manually, you need to install tkinter. "
                "See https://stackoverflow.com/a/74607246 for more information."
            )
            raise e
        root = tk.Tk()
        root.withdraw()
        user_input = simpledialog.askstring("Score Prompt", message).strip()
        root.destroy()
        return user_input.strip()
    
    def _validate_human_weight(self, weight: str) -> float:
        try:
            value = float(weight)
            if value < 0 or value > 1:
                raise ValueError("Weight must be between 0 and 1")
        except ValueError:
            raise ValueError(f"Weights require a numberic value. Got {weight}")
        return value
    
    async def validate(self, request_response, *, task = None):
        if request_response.original_value_data_type != "text":
            raise ValueError("The original value data type must be text.")
        if not task:
            raise ValueError("Task must be provided.")
    
    def _validate_scale_arguments_set(self, scale_args: dict):

        try:
            minimum_value = scale_args["minimum_value"]
            maximum_value = scale_args["maximum_value"]
            category = scale_args["category"]
            task = scale_args["task"]
        except KeyError as e:
            raise ValueError(f"Missing key in scale_args: {e.args[0]}") from None

        if not isinstance(minimum_value, int):
            raise ValueError(f"Minimum value must be an integer, got {type(minimum_value).__name__}.")
        if not isinstance(maximum_value, int):
            raise ValueError(f"Maximum value must be an integer, got {type(maximum_value).__name__}.")
        if minimum_value > maximum_value:
            raise ValueError("Minimum value must be less than or equal to the maximum value.")
        if not category:
            raise ValueError("Category must be set and cannot be empty.")
        if not task:
            raise ValueError("Task must be set and cannot be empty.")
        