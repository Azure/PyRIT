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
from pyrit.score import LikertScalePaths, ScorerEvalConfig, SelfAskLikertScorer


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.parametrize(
    "scale_path,config_file_name,metrics_file_name",
    [
        (LikertScalePaths.HATE_SPEECH_SCALE.value, "sample_config_likert_hate.yaml", "hate_speech_scale.json"),
        (LikertScalePaths.VIOLENCE_SCALE.value, "sample_config_likert_violence.yaml", "violence_scale.json"),
    ],
)
@pytest.mark.asyncio
async def test_likert_scorer_mae(scale_path: Path, config_file_name: str, metrics_file_name: str):
    metrics_file = Path(SCORER_EVALS_PATH, "metrics", "likert_scorer", metrics_file_name).resolve()
    with open(metrics_file, "r") as file:
        metrics = json.load(file)

    # using ~95% confidence interval for the mean absolute error (MAE) for thresholding
    mae_threshold = metrics["MAE"] + (2 * metrics["MAE_std_error"])

    target = OpenAIChatTarget(temperature=0.0, seed=1)
    likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=scale_path)

    config_path = Path(SCORER_EVALS_PATH, "eval_configs", config_file_name).resolve()
    eval_json = await likert_scorer.run_evaluation(config=ScorerEvalConfig.from_yaml_file(config_path), batch_size=10)
    eval_dict = json.loads(eval_json)
    assert eval_dict["MAE"] <= mae_threshold, f"MAE {eval_dict['MAE']} exceeds threshold {mae_threshold}"
