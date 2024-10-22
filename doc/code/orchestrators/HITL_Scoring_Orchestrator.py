# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to use HITL Scoring

# %%
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget
from pyrit.common import default_values

default_values.load_default_env()

# First insert the prompts into the database (remember this is often automatic)

prompts_to_score = ["I hate you.", "I hate all blondes"]

target = TextTarget()

with PromptSendingOrchestrator(prompt_target=target) as send_all_prompts_orchestrator:

    requests = await send_all_prompts_orchestrator.send_prompts_async(prompt_list=prompts_to_score)  # type: ignore
    prompt_sending_orchestrator_id = send_all_prompts_orchestrator.get_identifier()["id"]

# %%
# pylint: disable=W0611

# To start, we score the responses using an automated scorer

import time
from pyrit.memory import DuckDBMemory
from pyrit.orchestrator import ScoringOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    AzureContentFilterScorer,
    SelfAskCategoryScorer,
    HumanInTheLoopScorer,
    ContentClassifierPaths,
)

# we need the id from the previous run to score all prompts from the orchestrator
id = prompt_sending_orchestrator_id

# The scorer is interchangeable with other scorers
# scorer = AzureContentFilterScorer()

# EG SCORE PROMPTS 1st using automatic scorer
scorer = SelfAskCategoryScorer(
    chat_target=OpenAIChatTarget(), content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value
)

memory = DuckDBMemory()
scores = []
with ScoringOrchestrator() as scoring_orchestrator:
    start = time.time()
    scores = await scoring_orchestrator.score_prompts_by_orchestrator_id_async(  # type: ignore
        scorer=scorer, orchestrator_ids=[id], responses_only=False
    )
    end = time.time()

    print(f"Elapsed time for operation: {end-start}")

    for score in scores:
        prompt_text = memory.get_prompt_request_pieces_by_id(prompt_ids=[str(score.prompt_request_response_id)])[
            0
        ].original_value
        print(f"{score} : {prompt_text}")


# %%
# THEN CALL HITL SCORER to edit these scores
scorer = HumanInTheLoopScorer()  # PASS IN THE SCORER YOU WANT TO USE
with ScoringOrchestrator() as scoring_orchestrator:
    start = time.time()
    scores = await scoring_orchestrator.score_prompts_by_orchestrator_id_async(  # type: ignore
        scorer=scorer, orchestrator_ids=[id], responses_only=False
    )
    end = time.time()

    print(f"Elapsed time for operation: {end-start}")

    for score in scores:
        prompt_text = memory.get_prompt_request_pieces_by_id(prompt_ids=[str(score.prompt_request_response_id)])[
            0
        ].original_value
        print(f"{score} : {prompt_text}")

memory.dispose_engine()
