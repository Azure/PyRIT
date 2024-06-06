# %% [markdown]
# # Scoring Orchestrator
#
# A very common scenario is wanting to score multiple prompts. Here is a general strategy to do that using `ScoringOrchestrator`.
#
# 1. Get the `PromptRequestPiece`s into the database. This is done automatically if using any targets (e.g. running any of the demos), but even if you manually entered the prompts outside of PyRIT, you can import them using `TextTarget`s or csvs as described [here](../memory/memory.ipynb)
# 1. Scoring all prompts in the database that meet any criteria
#
# The following example demonstrates this by manually entering prompts in the database and then scoring them.

# %%
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget
from pyrit.common import default_values


default_values.load_default_env()


# First insert the prompts into the database (remember this is often automatic)

prompts_to_score = ["I'm sorry, I can't do that.", "I hate you.", "nothing wrong with this."]

target = TextTarget()

with PromptSendingOrchestrator(prompt_target=target) as send_all_prompts_orchestrator:

    requests = await send_all_prompts_orchestrator.send_prompts_async(prompt_list=prompts_to_score)  # type: ignore
    prompt_sending_orchestrator_id = int(send_all_prompts_orchestrator.get_identifier()["id"])



# %% [markdown]
# Once the prompts are in the database (which again, is often automatic) we can use `ScoringOrchestrator` to score them with whatever scorers we want. It works in parallel with batches.

# %%
# pylint: disable=W0611

from pyrit.memory import DuckDBMemory
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import AzureOpenAIChatTarget
from pyrit.score import (
    AzureContentFilterScorer,
    SelfAskCategoryScorer,
    HumanInTheLoopScorer,
    ContentClassifierPaths,
)

# we need the id from the previous run to score all prompts from the orchestrator
id = prompt_sending_orchestrator_id

# The scorer is interchangeable with other scorers
scorer = AzureContentFilterScorer()
# scorer = HumanInTheLoopScorer()
# scorer = SelfAskCategoryScorer(chat_target=AzureOpenAIChatTarget(), content_classifier=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value)

with ScoringOrchestrator() as scoring_orchestrator:
    scores = await scoring_orchestrator.score_prompts_by_orchestrator_id_async(  # type: ignore
        scorer=scorer, orchestrator_ids=[id], responses_only=False
    )

    memory = DuckDBMemory()

    for score in scores:
        prompt_text = memory.get_prompt_request_pieces_by_id(prompt_ids=[str(score.prompt_request_response_id)])[
            0
        ].original_value
        print(f"{score} : {prompt_text}")
