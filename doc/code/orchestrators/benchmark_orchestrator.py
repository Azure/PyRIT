# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Benchmark Orchestrator

# %%
from pyrit.orchestrator.question_answer_benchmark_orchestrator import QuestionAnsweringBenchmarkOrchestrator
from pyrit.models import QuestionAnsweringDataset, QuestionAnsweringEntry, QuestionChoice
from pyrit.prompt_target import AzureOpenAIGPT4OChatTarget
from pyrit.score.question_answer_scorer import QuestionAnswerScorer

from pyrit.common import default_values

default_values.load_default_env()

target = AzureOpenAIGPT4OChatTarget()

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

qa_scorer = QuestionAnswerScorer(
    dataset=qa_ds,
)

benchmark_orchestrator = QuestionAnsweringBenchmarkOrchestrator(
    chat_model_under_evaluation=target, scorer=qa_scorer, verbose=True
)

await benchmark_orchestrator.evaluate()  # type: ignore

# %%
correct_count = 0
total_count = 0

for idx, (qa_question_entry, answer) in enumerate(benchmark_orchestrator._scorer.evaluation_results.items()):
    print(f"Question {idx+1}: {qa_question_entry.question}")
    print(f"Answer: {answer}")
    print(f"")

    correct_count += 1 if answer.is_correct else 0

print(f"Correct count: {correct_count}/{len(benchmark_orchestrator._scorer.evaluation_results)}")
