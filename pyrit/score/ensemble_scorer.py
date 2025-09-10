import uuid
from typing import Optional, Dict, Literal, get_args
from dataclasses import dataclass

from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer

@dataclass
class WeakScorerSpec:
    scorer: Scorer
    weight: Optional[float] = None
    class_weights: Optional[Dict[str, float]] = None

LossMetric = Literal["MSE", "MAE"]

class EnsembleScorer(Scorer):

    def __init__(self, 
                 *,
                 weak_scorer_dict: Dict[str, WeakScorerSpec],
                 ground_truth_scorer: Scorer,
                 fit_weights: bool = False,
                 num_steps: int = 100,
                 lr: float = 1e-2,
                 category: str = "jailbreak"):
        self.scorer_type = "float_scale"
        self._score_category = category

        if not isinstance(weak_scorer_dict, dict) or (len(weak_scorer_dict) == 0):
            raise ValueError("Please pass a nonempty dictionary of weights")

        for scorer_name, weak_scorer_spec in weak_scorer_dict.items():
            if scorer_name == "AzureContentFilterScorer":
                if not isinstance(weak_scorer_spec.class_weights, dict) or len(weak_scorer_spec.class_weights) == 0:
                    raise ValueError("Weights for AzureContentFilterScorer must be a dictionary of category (str) to weight (float)")
                for acfs_k, acfs_v in weak_scorer_spec.class_weights.items():
                    if not isinstance(acfs_k, str) or not isinstance(acfs_v, float):
                        raise ValueError("Weights for AzureContentFilterScorer must be a dictionary of category (str) to weight (float)")
            elif not isinstance(weak_scorer_spec.weight, float):
                raise ValueError("Weight for this scorer must be a float")
            
        if not isinstance(lr, float) or lr <= 0:
            raise ValueError("Learning rate must be a floating point number greater than 0")

        self._weak_scorer_dict = weak_scorer_dict

        self._fit_weights = fit_weights
        self._num_steps_remaining = num_steps
        self._lr = lr

        self._ground_truth_scorer = ground_truth_scorer

    async def _score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self.validate(request_response, task=task)

        ensemble_score_value = 0
        ensemble_score_rationale = ""
        score_values = {}
        metadata = {}
        for scorer_name, weak_scorer_spec in self._weak_scorer_dict.items():
            scorer = weak_scorer_spec.scorer
            current_scores = await scorer.score_async(request_response=request_response, task=task)
            for curr_score in current_scores:
                if scorer_name == "AzureContentFilterScorer":
                    score_category = curr_score.score_category
                    curr_weight = weak_scorer_spec.class_weights[score_category]
                    metadata_label = "_".join([scorer_name, score_category, "weight"])

                    curr_score_value = float(curr_score.get_value())
                    if scorer_name not in score_values:
                        score_values[scorer_name] = {}
                    score_values[scorer_name][score_category] = curr_score_value

                    ensemble_score_rationale += f"{scorer_name}({score_category}) has value {curr_score_value} with weight {curr_weight}\n"
                else:
                    curr_weight = weak_scorer_spec.weight
                    metadata_label = "_".join([scorer_name, "weight"])
                    curr_score_value = float(curr_score.get_value())
                    score_values[scorer_name] = curr_score_value

                    ensemble_score_rationale += f"{scorer_name} has value {curr_score_value} with weight {curr_weight}\n"
                
                ensemble_score_value += curr_weight * curr_score_value

                metadata[metadata_label] = str(curr_weight)

        ensemble_score_rationale += f"Total Ensemble Score is {ensemble_score_value}"

        ensemble_score = Score(
            id=uuid.uuid4(),
            score_type="float_scale",
            score_value=str(ensemble_score_value),
            score_value_description=None,
            score_category=self._score_category,
            score_metadata=metadata,
            score_rationale=ensemble_score_rationale,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id,
            task=task,
        )

        if self._fit_weights and self._num_steps_remaining > 0:
            self._num_steps_remaining -= 1
            await self.step_weights(score_values=score_values, ensemble_score=ensemble_score, request_response=request_response, task=task)

        return [ensemble_score]

    async def step_weights(self, 
                           *,
                           score_values: Dict[str, float], 
                           ensemble_score: Scorer,
                           request_response: PromptRequestPiece, 
                           task: Optional[str] = None,
                           loss_metric: LossMetric = "MSE"):
        if loss_metric not in get_args(LossMetric):
            raise ValueError(f"Loss metric {loss_metric} is not a valid loss metric.")

        ground_truth_scores = await self._ground_truth_scorer.score_async(request_response=request_response, task=task)
        for ground_truth_score in ground_truth_scores:
            print(f"Ground Truth Score: {ground_truth_score.get_value()}")
            print(f"Ensemble Score: {ensemble_score.get_value()}")
            if loss_metric == "MSE":
                diff = ensemble_score.get_value() - float(ground_truth_score.get_value())
                d_loss_d_ensemble_score = 2 * diff
            elif loss_metric == "MAE":
                diff = ensemble_score.get_value() - float(ground_truth_score.get_value())
                if diff == 0:
                    d_loss_d_ensemble_score = 0
                elif diff < 0:
                    d_loss_d_ensemble_score = -1
                else:
                    d_loss_d_ensemble_score = 1


            for scorer_name in score_values:
                if scorer_name == "AzureContentFilterScorer":
                    self._weak_scorer_dict[scorer_name].class_weights = {score_category: 
                                                                            self._weak_scorer_dict[scorer_name].class_weights[score_category] -
                                                                            self._lr * score_values[scorer_name][score_category] * d_loss_d_ensemble_score
                                                                         for score_category in self._weak_scorer_dict[scorer_name].class_weights.keys()}
                else:
                    self._weak_scorer_dict[scorer_name].weight = self._weak_scorer_dict[scorer_name].weight - self._lr * score_values[scorer_name] * d_loss_d_ensemble_score
        
            print(f"Updated Weights: {self._weak_scorer_dict}")

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if request_response.original_value_data_type != "text":
            raise ValueError("The original value data type must be text.")
        if not task:
            raise ValueError("Task must be provided.")