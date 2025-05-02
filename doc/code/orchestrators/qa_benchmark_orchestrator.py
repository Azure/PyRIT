# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: pyrit-dev
# ---

# %% [markdown]
# # Q&A Benchmark Orchestrator - optional
#
# The `QuestionAnsweringBenchmarkOrchestrator` can process Q&A datasets and evaluate how good a target is at answering the questions.

# %%
# Import necessary packages
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_wmdp_dataset
from pyrit.memory import CentralMemory
from pyrit.models import (
    QuestionAnsweringDataset,
    QuestionAnsweringEntry,
    QuestionChoice,
)
from pyrit.orchestrator import QuestionAnsweringBenchmarkOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score.question_answer_scorer import QuestionAnswerScorer

# %%
# Initialize PyRIT (load environment files and set central memory instance)
initialize_pyrit(memory_db_type=IN_MEMORY)

# %%
# Set up the Azure OpenAI prompt target
target = OpenAIChatTarget()

# %%
# Create demo dataset for Q/A Model
qa_ds = QuestionAnsweringDataset(
    name="demo dataset",
    version="1.0",
    description="A dataset for testing the question answering model",
    author="PyRIT Team",
    group="",
    source="",
    questions=[
        QuestionAnsweringEntry(
            question="What is the capital of France?",
            answer_type="str",
            correct_answer="0",
            choices=[
                QuestionChoice(index=0, text="Paris"),
                QuestionChoice(index=1, text="London"),
                QuestionChoice(index=2, text="Berlin"),
                QuestionChoice(index=3, text="Madrid"),
            ],
        ),
        QuestionAnsweringEntry(
            question="What is the capital of Germany?",
            answer_type="str",
            correct_answer="2",
            choices=[
                QuestionChoice(index=0, text="Paris"),
                QuestionChoice(index=1, text="London"),
                QuestionChoice(index=2, text="Berlin"),
                QuestionChoice(index=3, text="Madrid"),
            ],
        ),
        QuestionAnsweringEntry(
            question="Which of the following is a prime number?",
            answer_type="str",
            correct_answer="3",
            choices=[
                QuestionChoice(index=0, text="507963"),
                QuestionChoice(index=1, text="507962"),
                QuestionChoice(index=2, text="507960"),
                QuestionChoice(index=3, text="507961"),
            ],
        ),
    ],
)

# Create the scorer
qa_scorer = QuestionAnswerScorer()

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator = QuestionAnsweringBenchmarkOrchestrator(
    objective_target=target, scorers=[qa_scorer], verbose=True
)

# Sends prompts associated with dataset
responses = await benchmark_orchestrator.send_prompts_async(dataset=qa_ds)  # type: ignore

# %%
qa_scorer.report_scores(responses=responses)

# %%
# Fetch WMDP dataset for Q/A Model Testing

wmdp_ds = fetch_wmdp_dataset()
wmdp_ds.questions = wmdp_ds.questions[:3]

# Evaluate the Q/A Model response
responses = await benchmark_orchestrator.send_prompts_async(dataset=wmdp_ds)  # type: ignore

# %%
# Output if the results are correct
qa_scorer.report_scores(responses=responses)

# %%
# Fetch WMDP dataset for Q/A Model Testing - Chem Subset

wmdp_ds = fetch_wmdp_dataset(category="chem")
wmdp_ds.questions = wmdp_ds.questions[:3]

# Evaluate the Q/A Model response
responses = await benchmark_orchestrator.send_prompts_async(dataset=wmdp_ds)  # type: ignore

# %%
# Output if the results are correct
qa_scorer.report_scores(responses=responses)

# %%
# Fetch WMDP dataset for Q/A Model Testing - Bio Subset

wmdp_ds = fetch_wmdp_dataset(category="bio")
wmdp_ds.questions = wmdp_ds.questions[:3]

# Evaluate the Q/A Model response
responses = await benchmark_orchestrator.send_prompts_async(dataset=wmdp_ds)  # type: ignore

# %%
# Output if the results are correct
qa_scorer.report_scores(responses=responses)

# %%
# Fetch WMDP dataset for Q/A Model Testing - Cyber Subset

wmdp_ds = fetch_wmdp_dataset(category="cyber")
wmdp_ds.questions = wmdp_ds.questions[:3]

# Evaluate the Q/A Model response
responses = await benchmark_orchestrator.send_prompts_async(dataset=wmdp_ds)  # type: ignore

# %%
# Output if the results are correct
qa_scorer.report_scores(responses=responses)

# %%
# Close connection for memory instance
memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
