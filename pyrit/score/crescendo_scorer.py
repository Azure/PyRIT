import json
import uuid

from pathlib import Path
from typing import Union

from pyrit.interfaces import SupportTextClassification
from pyrit.models import Score, ScorerStrategy
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import PromptChatTarget


# required input: conversation_objective, response_to_evaluate_input
# output: result_flag, result_percentage, result_explanation
EVALUATOR = Path(__file__).parent / "crescendo" / "eval_prompt.yaml"

# required input: result_explanation
# output: second_layer_result_flag, second_layer_result_explanation
META_JUDGE_PROMPT = Path(__file__).parent / "crescendo" / "meta_judge_prompt.yaml"

# required input: conversation_objective, response_to_evaluate_input
# output: result_flag, result_percentage, result_explanation
REFUSE_JUDGE_PROMPT = Path(__file__).parent / "crescendo" / "refuse_judge_prompt.yaml"


class CrescendoScorer(SupportTextClassification):
    def __init__(self, prompt: Union[str, Path], chat_target: PromptChatTarget, **kwargs) -> None:
        # Load classifier
        self._chat_target = chat_target

        self._conversation_id = str(uuid.uuid4())
        self._normalizer_id = None  # Normalizer not used

        self.labels = {"scorer": "crescendo_scorer"}

        self.scorer_strategy = ScorerStrategy(strategy=prompt, **kwargs)

        self._chat_target.set_system_prompt(
            system_prompt=str(self.scorer_strategy),
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
            labels=self.labels,
        )

    # @tenacity.retry(stop=tenacity.stop_after_attempt(3))
    def score_text(self, text: str) -> Score:

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_prompt_text=text,
                    conversation_id=self._conversation_id,
                    labels=self.labels,
                    prompt_target_identifier=self._chat_target.get_identifier(),
                )
            ]
        )

        response_text = self._chat_target.send_prompt(prompt_request=request).request_pieces[0].converted_prompt_text

        try:
            parsed_response = json.loads(response_text)

            score = Score(
                score_type="bool",
                score_value=(
                    parsed_response["result_flag"]
                    if "result_flag" in parsed_response
                    else parsed_response["second_layer_result_flag"]
                ),
                score_description=(
                    parsed_response["result_percentage"] if "result_percentage" in parsed_response else None
                ),
                score_explanation=(
                    parsed_response["result_explanation"]
                    if "result_explanation" in parsed_response
                    else parsed_response["second_layer_result_explanation"]
                ),
            )

            return score

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from chat target: {response_text}") from e
