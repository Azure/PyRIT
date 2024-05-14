import json
import uuid

from pathlib import Path
from typing import Union

from pyrit.score import Score, Scorer
from pyrit.score.scorer_strategy import ScorerStrategy
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


class CrescendoScorer(Scorer):
    def __init__(self, prompt: Union[str, Path], chat_target: PromptChatTarget) -> None:
        # Load classifier
        self._chat_target = chat_target

        self._conversation_id = str(uuid.uuid4())
        self._normalizer_id = None  # Normalizer not used

        self.labels = {"scorer": "crescendo_scorer"}

        self.prompt = prompt

        self._chat_target.set_system_prompt(
            system_prompt="You are a helpful assistant.",
            conversation_id=self._conversation_id,
            orchestrator_identifier=None,
            labels=self.labels,
        )

    # @tenacity.retry(stop=tenacity.stop_after_attempt(3))
    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:

        self.validate(request_response)

        # get all the key value pairs from the request_response.metadata json and pass them onto the
        # ScorerStrategy kwargs
        kwargs = {}
        if request_response.prompt_metadata:
            metadata = json.loads(request_response.prompt_metadata)
            kwargs = metadata

        self.scorer_strategy = ScorerStrategy(strategy=self.prompt, **kwargs)

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=str(self.scorer_strategy),
                    converted_value=str(self.scorer_strategy),
                    conversation_id=self._conversation_id,
                    labels=self.labels,
                    prompt_target_identifier=self._chat_target.get_identifier(),
                )
            ]
        )

        response_text = self._chat_target.send_prompt(prompt_request=request).request_pieces[0].converted_value

        try:
            parsed_response = json.loads(response_text)

            score = Score(
                score_type="true_false",
                score_value=(
                    parsed_response["result_flag"]
                    if "result_flag" in parsed_response
                    else parsed_response["second_layer_result_flag"]
                ),
                score_value_description="result_flag",
                score_category=None,
                score_metadata=(
                    "result_percentage=" + str(parsed_response["result_percentage"])
                    if "result_percentage" in parsed_response
                    else ""
                ),
                score_rationale=(
                    parsed_response["result_explanation"]
                    if "result_explanation" in parsed_response
                    else parsed_response["second_layer_result_explanation"]
                ),
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
            )

            return [score]

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from chat target: {response_text}") from e

    def validate(self, request_response: PromptRequestPiece):
        if request_response.converted_value_data_type != "text":
            raise ValueError("Crescendo scorer only supports text data type")
