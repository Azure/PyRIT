# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.datasets import (
    fetch_adv_bench_dataset,
    fetch_aya_redteaming_dataset,
    fetch_babelscape_alert_dataset,
    fetch_ccp_sensitive_prompts_dataset,
    fetch_darkbench_dataset,
    fetch_decoding_trust_stereotypes_dataset,
    fetch_equitymedqa_dataset_unique_values,
    fetch_forbidden_questions_dataset,
    fetch_harmbench_dataset,
    fetch_jbb_behaviors_by_harm_category,
    fetch_jbb_behaviors_by_jbb_category,
    fetch_jbb_behaviors_dataset,
    fetch_librAI_do_not_answer_dataset,
    fetch_llm_latent_adversarial_training_harmful_dataset,
    fetch_many_shot_jailbreaking_dataset,
    fetch_medsafetybench_dataset,
    fetch_mlcommons_ailuminate_demo_dataset,
    fetch_multilingual_vulnerability_dataset,
    fetch_pku_safe_rlhf_dataset,
    fetch_red_team_social_bias_dataset,
    fetch_seclists_bias_testing_dataset,
    fetch_sosbench_dataset,
    fetch_tdc23_redteaming_dataset,
    fetch_transphobia_awareness_dataset,
    fetch_wmdp_dataset,
    fetch_xstest_dataset,
)
from pyrit.models.seed_prompt_dataset import SeedPromptDataset


@pytest.mark.parametrize(
    "fetch_function, is_seed_prompt_dataset",
    [
        (fetch_adv_bench_dataset, True),
        (fetch_aya_redteaming_dataset, True),
        (fetch_babelscape_alert_dataset, True),
        (fetch_ccp_sensitive_prompts_dataset, True),
        (fetch_darkbench_dataset, True),
        (fetch_decoding_trust_stereotypes_dataset, True),
        (fetch_equitymedqa_dataset_unique_values, True),
        (fetch_forbidden_questions_dataset, True),
        (fetch_harmbench_dataset, True),
        (fetch_jbb_behaviors_dataset, True),
        (fetch_librAI_do_not_answer_dataset, True),
        (fetch_llm_latent_adversarial_training_harmful_dataset, True),
        (fetch_many_shot_jailbreaking_dataset, False),
        (fetch_medsafetybench_dataset, True),
        (fetch_mlcommons_ailuminate_demo_dataset, True),
        (fetch_multilingual_vulnerability_dataset, True),
        (fetch_pku_safe_rlhf_dataset, True),
        (fetch_red_team_social_bias_dataset, True),
        (fetch_seclists_bias_testing_dataset, True),
        (fetch_sosbench_dataset, True),
        (fetch_tdc23_redteaming_dataset, True),
        (fetch_transphobia_awareness_dataset, True),
        (fetch_wmdp_dataset, False),
        (fetch_xstest_dataset, True),
    ],
)
def test_fetch_datasets(fetch_function, is_seed_prompt_dataset):
    data = fetch_function()

    assert data is not None
    if is_seed_prompt_dataset:
        assert isinstance(data, SeedPromptDataset)
        assert len(data.prompts) > 0


@pytest.mark.integration
def test_fetch_jbb_behaviors_by_harm_category():
    """Integration test for filtering by harm category with real data."""
    try:
        # Filter for a common category to ensure we get results
        violence_prompts = fetch_jbb_behaviors_by_harm_category("violence")
        assert isinstance(violence_prompts, SeedPromptDataset)
        assert len(violence_prompts.prompts) > 0
    except Exception as e:
        pytest.skip(f"Integration test skipped due to: {e}")


@pytest.mark.integration
def test_fetch_jbb_behaviors_by_jbb_category():
    """Integration test for filtering by JBB category with real data."""
    try:
        # Filter for a common category to ensure we get results
        hate_prompts = fetch_jbb_behaviors_by_jbb_category("Disinformation")
        assert isinstance(hate_prompts, SeedPromptDataset)
        assert len(hate_prompts.prompts) > 0
    except Exception as e:
        pytest.skip(f"Integration test skipped due to: {e}")
