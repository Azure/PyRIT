# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Benchmark Orchestrator - optional

# %%
# Import necessary packages
from pyrit.orchestrator.question_answer_benchmark_orchestrator import QuestionAnsweringBenchmarkOrchestrator
from pyrit.models import QuestionAnsweringDataset, QuestionAnsweringEntry, QuestionChoice
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score.question_answer_scorer import QuestionAnswerScorer
from pyrit.datasets import fetch_wmdp_dataset
from pyrit.common import default_values

# %%
# Load environment variables
default_values.load_environment_files()

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
            correct_answer="Paris",
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
            correct_answer="Berlin",
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
            correct_answer="507961",
            choices=[
                QuestionChoice(index=0, text="507963"),
                QuestionChoice(index=1, text="507962"),
                QuestionChoice(index=2, text="507960"),
                QuestionChoice(index=3, text="507961"),
            ],
        ),
    ],
)

# Create the score for the Q/A Model
qa_scorer = QuestionAnswerScorer(
    dataset=qa_ds,
)

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator = QuestionAnsweringBenchmarkOrchestrator(
    chat_model_under_evaluation=target, scorer=qa_scorer, verbose=True
)

# Evaluate the Q/A Model response
await benchmark_orchestrator.evaluate()  # type: ignore

# %%
# Output if the results are correct
correct_count = 0
total_count = 0

for idx, (qa_question_entry, answer) in enumerate(benchmark_orchestrator._scorer.evaluation_results.items()):
    print(f"Question {idx+1}: {qa_question_entry.question}")
    print(f"Answer: {answer}")
    print(f"")

    correct_count += 1 if answer.is_correct else 0

print(f"Correct count: {correct_count}/{len(benchmark_orchestrator._scorer.evaluation_results)}")

# %%
# Fetch WMDP dataset for Q/A Model Testing

wmdp_ds = fetch_wmdp_dataset()
wmdp_ds.questions = wmdp_ds.questions[:3]

# Create the score for the Q/A Model
qa_scorer_wmdp = QuestionAnswerScorer(
    dataset=wmdp_ds,
)

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator_wmdp = QuestionAnsweringBenchmarkOrchestrator(
    chat_model_under_evaluation=target, scorer=qa_scorer_wmdp, verbose=True
)

# Evaluate the Q/A Model response
await benchmark_orchestrator_wmdp.evaluate()  # type: ignore

# %%
# Output if the results are correct
correct_count = 0
total_count = 0

for idx, (qa_question_entry, answer) in enumerate(benchmark_orchestrator_wmdp._scorer.evaluation_results.items()):
    print(f"Question {idx+1}: {qa_question_entry.question}")
    print(f"Answer: {answer}")
    print(f"")

    correct_count += 1 if answer.is_correct else 0

print(f"Correct count: {correct_count}/{len(benchmark_orchestrator_wmdp._scorer.evaluation_results)}")

# %%
# Fetch WMDP dataset for Q/A Model Testing - Chem Subset

wmdp_ds = fetch_wmdp_dataset(category="chem")
wmdp_ds.questions = wmdp_ds.questions[:3]

# Create the score for the Q/A Model
qa_scorer_wmdp = QuestionAnswerScorer(
    dataset=wmdp_ds,
)

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator_wmdp = QuestionAnsweringBenchmarkOrchestrator(
    chat_model_under_evaluation=target, scorer=qa_scorer_wmdp, verbose=True
)

# Evaluate the Q/A Model response
await benchmark_orchestrator_wmdp.evaluate()  # type: ignore

# %%
# Output if the results are correct
correct_count = 0
total_count = 0

for idx, (qa_question_entry, answer) in enumerate(benchmark_orchestrator_wmdp._scorer.evaluation_results.items()):
    print(f"Question {idx+1}: {qa_question_entry.question}")
    print(f"Answer: {answer}")
    print(f"")

    correct_count += 1 if answer.is_correct else 0

print(f"Correct count: {correct_count}/{len(benchmark_orchestrator_wmdp._scorer.evaluation_results)}")

# %%
# Fetch WMDP dataset for Q/A Model Testing - Bio Subset

wmdp_ds = fetch_wmdp_dataset(category="bio")
wmdp_ds.questions = wmdp_ds.questions[:3]

# Create the score for the Q/A Model
qa_scorer_wmdp = QuestionAnswerScorer(
    dataset=wmdp_ds,
)

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator_wmdp = QuestionAnsweringBenchmarkOrchestrator(
    chat_model_under_evaluation=target, scorer=qa_scorer_wmdp, verbose=True
)

# Evaluate the Q/A Model response
await benchmark_orchestrator_wmdp.evaluate()  # type: ignore

# %%
# Output if the results are correct
correct_count = 0
total_count = 0

for idx, (qa_question_entry, answer) in enumerate(benchmark_orchestrator_wmdp._scorer.evaluation_results.items()):
    print(f"Question {idx+1}: {qa_question_entry.question}")
    print(f"Answer: {answer}")
    print(f"")

    correct_count += 1 if answer.is_correct else 0

print(f"Correct count: {correct_count}/{len(benchmark_orchestrator_wmdp._scorer.evaluation_results)}")

# %%
# Fetch WMDP dataset for Q/A Model Testing - Cyber Subset

wmdp_ds = fetch_wmdp_dataset(category="cyber")
wmdp_ds.questions = wmdp_ds.questions[:3]

# Create the score for the Q/A Model
qa_scorer_wmdp = QuestionAnswerScorer(
    dataset=wmdp_ds,
)

# Create the orchestrator with scorer and demo dataset
benchmark_orchestrator_wmdp = QuestionAnsweringBenchmarkOrchestrator(
    chat_model_under_evaluation=target, scorer=qa_scorer_wmdp, verbose=True
)

# Evaluate the Q/A Model response
await benchmark_orchestrator_wmdp.evaluate()  # type: ignore

# %%
# Output if the results are correct
correct_count = 0
total_count = 0

for idx, (qa_question_entry, answer) in enumerate(benchmark_orchestrator_wmdp._scorer.evaluation_results.items()):
    print(f"Question {idx+1}: {qa_question_entry.question}")
    print(f"Answer: {answer}")
    print(f"")

    correct_count += 1 if answer.is_correct else 0

print(f"Correct count: {correct_count}/{len(benchmark_orchestrator_wmdp._scorer.evaluation_results)}")

# %%
try:
    wmdp_ds = fetch_wmdp_dataset(category="invalid string")
except ValueError:
    print("TEST PASS. Value Error Caught.")
else:
    print("TEST FAILED. No ValueError Caught.")
