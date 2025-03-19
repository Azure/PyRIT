# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_forbidden_questions_dataset() -> SeedPromptDataset:
    """
    Fetch Forbidden question dataset and return it as a SeedPromptDataset

    Returns: SeedPromptDataset

    Note: For more info
        Paper - https://arxiv.org/abs/2308.03825
        Github - https://github.com/verazuo/jailbreak_llms/
        Website - https://jailbreak-llms.xinyueshen.me/
    """
    data = load_dataset("TrustAIRLab/forbidden_question_set", "default")

    authors = ["Xinyue Shen", "Zeyuan Chen", "Michael Backes", "Yun Shen", "Yang Zhang"]
    seed_prompts = [
        SeedPrompt(
            value=item["question"],
            data_type="text",
            name="TrustAIRLab/forbidden_question_set",
            dataset_name="TrustAIRLab/forbidden_question_set",
            authors=authors,
            harm_categories=item["content_policy_name"],
            source="https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set",
            description="""This is the Forbidden Question Set dataset proposed in the ACM CCS 2024 paper
        "Do Anything Now'': Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models.
        It contains 390 questions (= 13 scenarios x 30 questions) adopted from OpenAI Usage Policy.
        The focus is on 13 scenarios, including Illegal Activity, Hate Speech, Malware Generation,
        Physical Harm, Economic Harm, Fraud, Pornography, Political Lobbying, Privacy Violence, Legal Opinion,
        Financial Advice, Health Consultation, and Government Decision.""",
        )
        for item in data["train"]
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset
