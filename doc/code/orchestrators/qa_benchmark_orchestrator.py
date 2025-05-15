# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Q&A Benchmark Orchestrator - optional
#
# The `QuestionAnsweringBenchmarkOrchestrator` can process Q&A datasets and evaluate how good a target is at answering the questions.

# %%
# Import necessary packages
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_wmdp_dataset
from pyrit.models import (
    QuestionAnsweringEntry,
    QuestionChoice,
)
from pyrit.orchestrator import QuestionAnsweringBenchmarkOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score.self_ask_question_answer_scorer import SelfAskQuestionAnswerScorer

# Initialize PyRIT (load environment files and set central memory instance)
initialize_pyrit(memory_db_type=IN_MEMORY)

# %%
# Set up the Azure OpenAI prompt target

objective_scorer = SelfAskQuestionAnswerScorer(chat_target=OpenAIChatTarget())

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator = QuestionAnsweringBenchmarkOrchestrator(
    objective_target=OpenAIChatTarget(),
    objective_scorer=objective_scorer,
)

question = QuestionAnsweringEntry(
    question="What is the capital of France?",
    answer_type="str",
    correct_answer="0",
    choices=[
        QuestionChoice(index=0, text="Paris"),
        QuestionChoice(index=1, text="London"),
        QuestionChoice(index=2, text="Berlin"),
        QuestionChoice(index=3, text="Madrid"),
    ],
)


# Sends prompts associated with dataset
result = await benchmark_orchestrator.run_attack_async(question_answering_entry=question)  # type: ignore
await result.print_conversation_async()  # type: ignore

# %%
# Fetch WMDP dataset for Q/A Model Testing; this is big and can take a minute

wmdp_ds = fetch_wmdp_dataset(category="cyber")

# %%
# Evaluate the Q/A Model response
results = await benchmark_orchestrator.run_attacks_async(question_answering_entries=wmdp_ds.questions[:3])  # type: ignore
for result in results:
    await result.print_conversation_async()  # type: ignore

# %% [markdown]
# You can also do analysis and benchmarking using these . To see the percentage of correct answers, you can use the `OrchestratorResult` object.

# %%
success = sum(r.status == "success" for r in results)
print(f"Success rate: {success / len(results)}")
