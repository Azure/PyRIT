# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from typing import Generator

import pytest
from integration.mocks import get_memory_interface

from pyrit.common.path import SCORER_EVALS_PATH
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import LikertScalePaths, ScorerEvaluator, SelfAskLikertScorer


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
async def test_likert_scorer_mae():
    target = OpenAIChatTarget()
    scale_path = LikertScalePaths.HATE_SPEECH_SCALE.value
    likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=scale_path)

    evaluator = ScorerEvaluator(scorer=likert_scorer)
    csv_path = f"{str(SCORER_EVALS_PATH)}/eval_datasets/likert_hate_speech_sample_dataset.csv"

    baseline_metrics = likert_scorer.get_scorer_metrics(file_name="hate_speech")

    # using ~95% confidence interval for the mean absolute error (MAE) for thresholding
    mae_threshold = baseline_metrics.mean_absolute_error + (2 * baseline_metrics.mae_standard_error)

    target = OpenAIChatTarget(temperature=0.0, seed=1)
    likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=scale_path)

    metrics = await evaluator.run_evaluation_from_csv_async(
        csv_path=csv_path,
        type="harm",
        assistant_response_col="assistant_response",
        gold_label_col_names=["human_likert_score_1", "human_likert_score_2", "human_likert_score_3"],
        top_level_harm="hate_speech",
        save_results=False,
    )

    assert (
        metrics.mean_absolute_error <= mae_threshold
    ), f"MAE {metrics.mean_absolute_error} exceeds threshold {mae_threshold}"
