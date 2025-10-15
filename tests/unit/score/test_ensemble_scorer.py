import uuid
import os
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import (
    get_audio_request_piece,
    get_image_request_piece,
    get_test_request_piece,
)

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import Score

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.score import EnsembleScorer, WeakScorerSpec, SelfAskScaleScorer, AzureContentFilterScorer

@pytest.fixture
def audio_request_piece() -> PromptRequestPiece:
    return get_audio_request_piece()


@pytest.fixture
def image_request_piece() -> PromptRequestPiece:
    return get_image_request_piece()


@pytest.fixture
def text_request_piece() -> PromptRequestPiece:
    return get_test_request_piece()

@pytest.fixture
def scorer_scale_response() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"score_value": "1",
         "rationale": "rationale",
         "description": "description"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])

def create_ensemble_scorer(self_ask_scale_score_value, 
                           self_ask_scale_weight,
                           azure_content_filter_score_values,
                           azure_content_filter_weights,
                           ensemble_category = 'jailbreak',
                           ground_truth_score: float = 0.1,
                           lr: float = 1e-2) -> EnsembleScorer:
    self_ask_scale_objective_scorer = AsyncMock()
    self_ask_scale_objective_scorer.score_async = AsyncMock(
        return_value=[
            Score(
                score_value=str(self_ask_scale_score_value),
                score_type="float_scale",
                score_category="mock category",
                score_rationale="A mock rationale",
                score_metadata=None,
                prompt_request_response_id=uuid.uuid4(),
                score_value_description="A mock description",
                id=uuid.uuid4(),
            )
        ]
    )
    azure_content_filter_objective_scorer = AsyncMock()
    azure_content_filter_objective_scorer.score_async = AsyncMock(
        return_value=[
            Score(
                score_value=str(score_value),
                score_type="float_scale",
                score_category=category,
                score_rationale="A mock hate rationale",
                score_metadata=None,
                prompt_request_response_id=uuid.uuid4(),
                score_value_description="A mock hate description",
                id=uuid.uuid4(),
            )
            for category, score_value in azure_content_filter_score_values.items()
        ]
    )

    weak_scorer_dict = {"SelfAskScaleScorer": WeakScorerSpec(self_ask_scale_objective_scorer, 
                                                             self_ask_scale_weight), 
                        "AzureContentFilterScorer": WeakScorerSpec(azure_content_filter_objective_scorer, 
                                                                   {k: v for k,v in azure_content_filter_weights.items()})}
    
    ground_truth_scorer = MagicMock()
    ground_truth_scorer.score_async = AsyncMock(
        return_value=[
            Score(
                score_value=str(ground_truth_score),
                score_type="float_scale",
                score_category="mock ground truth category",
                score_rationale="A mock ground truth rationale",
                score_metadata=None,
                prompt_request_response_id=uuid.uuid4(),
                score_value_description="A mock ground truth description",
                id=uuid.uuid4(),
            )
        ]
    )
    
    scorer = EnsembleScorer(weak_scorer_dict=weak_scorer_dict,
                            ground_truth_scorer=ground_truth_scorer,
                            category=ensemble_category,
                            lr=lr)
    
    return scorer

@pytest.mark.asyncio
async def test_ensemble_scorer_validate_audio(audio_request_piece: PromptRequestPiece):
    self_ask_scale_score_value = 0.4
    self_ask_scale_score_weight = 0.8
    azure_content_filter_values = {"Hate": 0.05,
                                   "Violence": 0.05}
    azure_content_filter_weights = {"Hate": 0.1,
                                    "Violence": 0.1}

    scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                    self_ask_scale_score_weight,
                                    azure_content_filter_values,
                                    azure_content_filter_weights)
    with pytest.raises(ValueError, match="The original value data type must be text."):
        await scorer.validate(audio_request_piece)

    os.remove(audio_request_piece.converted_value)

@pytest.mark.asyncio
async def test_ensemble_scorer_validate_image(image_request_piece: PromptRequestPiece):
    self_ask_scale_score_value = 0.4
    self_ask_scale_score_weight = 0.8
    azure_content_filter_values = {"Hate": 0.05,
                                   "Violence": 0.05}
    azure_content_filter_weights = {"Hate": 0.1,
                                    "Violence": 0.1}

    scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                    self_ask_scale_score_weight,
                                    azure_content_filter_values,
                                    azure_content_filter_weights)
    with pytest.raises(ValueError, match="The original value data type must be text."):
        await scorer.validate(image_request_piece)

    os.remove(image_request_piece.converted_value)

@pytest.mark.asyncio
async def test_ensemble_scorer_validate_text(text_request_piece: PromptRequestPiece):
    scorer = create_ensemble_scorer()
    # should not raise an error
    scorer.validate(text_request_piece)

    os.remove(text_request_piece.converted_value)

@pytest.mark.asyncio
async def test_ensemble_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        self_ask_scale_score_value = 0.4
        self_ask_scale_score_weight = 0.8
        azure_content_filter_values = {"Hate": 0.05,
                                    "Violence": 0.05}
        azure_content_filter_weights = {"Hate": 0.1,
                                        "Violence": 0.1}

        scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                        self_ask_scale_score_weight,
                                        azure_content_filter_values,
                                        azure_content_filter_weights)
        await scorer.score_text_async(text="I hate you!")

        memory.add_scores_to_memory.assert_called_once()

@pytest.mark.asyncio
async def test_ensemble_scorer_score():
    self_ask_scale_score_value = 0.4
    self_ask_scale_score_weight = 0.8
    azure_content_filter_values = {"Hate": 0.05,
                                   "Violence": 0.05}
    azure_content_filter_weights = {"Hate": 0.1,
                                    "Violence": 0.1}

    scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                    self_ask_scale_score_weight,
                                    azure_content_filter_values,
                                    azure_content_filter_weights)
    score = await scorer.score_text_async(text="example text", task="example task")

    assert len(scorer) == 1

    true_ensemble_score = self_ask_scale_score_value * self_ask_scale_score_weight
    for azure_category in azure_content_filter_values:
        true_ensemble_score += azure_content_filter_values[azure_category] * azure_content_filter_weights[azure_category]

    assert score[0].score_value == true_ensemble_score
    assert score[0].score_value_description is None
    assert score[0].score_type == "float_scale"
    assert score[0].score_category == "jailbreak"
    assert score[0].score_rationale == f"Total Ensemble Score is {true_ensemble_score}"
    assert "EnsembleScorer" in str(score[0].scorer_class_identifier)

@pytest.mark.asyncio
async def test_ensemble_scorer_invalid_learning_rate():
    learning_rate = -1.1

    self_ask_scale_score_value = 0.4
    self_ask_scale_score_weight = 0.8
    azure_content_filter_values = {"Hate": 0.05,
                                   "Violence": 0.05}
    azure_content_filter_weights = {"Hate": 0.1,
                                    "Violence": 0.1}
    with pytest.raises(ValueError, match="Learning rate must be a floating point number greater than 0"):
        scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                        self_ask_scale_score_weight,
                                        azure_content_filter_values,
                                        azure_content_filter_weights,
                                        lr=learning_rate)
        
@pytest.mark.asyncio
async def test_ensemble_scorer_invalid_weights_azure_content_filter():
    azure_content_filter_scorer = AzureContentFilterScorer()
    weak_scorer_dict = {"AzureContentFilterScorer": WeakScorerSpec(azure_content_filter_scorer, 0.1)}

    ground_truth_scorer = MagicMock()
    with pytest.raises(ValueError, match="Weights for AzureContentFilterScorer must be a dictionary of category (str) to weight (float)"):
        scorer = EnsembleScorer(weak_scorer_dict=weak_scorer_dict,
                                ground_truth_scorer=ground_truth_scorer)
        
@pytest.mark.asyncio
async def test_ensemble_scorer_invalid_weight_non_azure_content_filter():
    chat_target = MagicMock()
    self_ask_scale_scorer = SelfAskScaleScorer(chat_target=chat_target)
    weak_scorer_dict = {"SelfAskScaleScorer": WeakScorerSpec(self_ask_scale_scorer, True)}

    ground_truth_scorer = MagicMock()
    with pytest.raises(ValueError, match="Weight for this scorer must be a float"):
        scorer = EnsembleScorer(weak_scorer_dict=weak_scorer_dict,
                                ground_truth_scorer=ground_truth_scorer)

@pytest.mark.parametrize("loss", ["MAE", "MSE"])
@pytest.mark.asyncio
async def test_ensemble_scorer_step(loss, scorer_scale_response):
    self_ask_scale_score_value = 0.4
    self_ask_scale_score_weight = 0.8
    azure_content_filter_values = {"Hate": 0.05,
                                   "Violence": 0.05}
    azure_content_filter_weights = {"Hate": 0.1,
                                    "Violence": 0.1}
    score_values = {"SelfAskScaleScorer": 0.4,
                    "AzureContentFilterScorer": {"Hate": 0.05, "Violence": 0.05}}
    ground_truth_score = 0.3
    lr = 1e-2

    true_ensemble_score = self_ask_scale_score_value * self_ask_scale_score_weight
    for azure_category in azure_content_filter_values:
        true_ensemble_score += azure_content_filter_values[azure_category] * azure_content_filter_weights[azure_category]

    scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                    self_ask_scale_score_weight,
                                    azure_content_filter_values,
                                    azure_content_filter_weights,
                                    ground_truth_score,
                                    lr = 1e-2)
    score = await scorer.score_text_async(text="example text", task="example task")
    
    
    await scorer.step_weights(score_values=score_values,
                              ensemble_score=score.get_value(),
                              lr=lr,
                              loss_metric=loss,
                              request_response=scorer_scale_response)
    
    if loss == "MSE":
        assert scorer._weak_scorer_dict["SelfAskScaleScorer"].weight == 0.8 - lr * 2 * (score.get_value() - ground_truth_score) * 0.4
        assert scorer._weak_scorer_dict["AzureContentFilterScorer"].class_weights["Hate"] == 0.1 - lr * 2 *(score.get_value() - ground_truth_score) * 0.05
        assert scorer._weak_scorer_dict["AzureContentFilterScorer"].class_weights["Violence"] == 0.1 - lr * 2 * (score.get_value() - ground_truth_score) * 0.05
    elif loss == "MAE":
        assert scorer._weak_scorer_dict["SelfAskScaleScorer"].weight == 0.8 - lr * ((score.get_value() - ground_truth_score) > 0) * 0.4
        assert scorer._weak_scorer_dict["AzureContentFilterScorer"].class_weights["Hate"] == 0.1 - lr * ((score.get_value() - ground_truth_score) > 0) * 0.05
        assert scorer._weak_scorer_dict["AzureContentFilterScorer"].class_weights["Violence"] == 0.1 - lr * ((score.get_value() - ground_truth_score) > 0) * 0.05

@pytest.mark.asyncio
async def test_ensemble_scorer_invalid_loss_metric(scorer_scale_response):
    loss_metric = "cosine similarity"

    self_ask_scale_score_value = 0.4
    self_ask_scale_score_weight = 0.8
    azure_content_filter_values = {"Hate": 0.05,
                                   "Violence": 0.05}
    azure_content_filter_weights = {"Hate": 0.1,
                                    "Violence": 0.1}
    score_values = {"SelfAskScaleScorer": 0.4,
                    "AzureContentFilterScorer": {"Hate": 0.05, "Violence": 0.05}}
    scorer = create_ensemble_scorer(self_ask_scale_score_value,
                                    self_ask_scale_score_weight,
                                    azure_content_filter_values,
                                    azure_content_filter_weights)

    with pytest.raises(ValueError, match=f"Loss metric {loss_metric} is not a valid loss metric."):
        scorer.step_weights(score_values=score_values,
                            ensemble_score=0.1,
                            loss_metric=loss_metric,
                            request_response=scorer_scale_response)

