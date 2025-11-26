# %%NBQA-CELL-SEP52c935
from pyrit.datasets import fetch_wmdp_dataset
from pyrit.executor.attack import AttackScoringConfig, ConsoleAttackResultPrinter
from pyrit.executor.benchmark import QuestionAnsweringBenchmark
from pyrit.models import (
    QuestionAnsweringEntry,
    QuestionChoice,
)
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskQuestionAnswerScorer
from pyrit.setup import IN_MEMORY, initialize_pyrit

# Initialize PyRIT (load environment files and set central memory instance)
initialize_pyrit(memory_db_type=IN_MEMORY)

# Define the scoring configuration for the benchmark
scoring_config = AttackScoringConfig(objective_scorer = SelfAskQuestionAnswerScorer(chat_target=OpenAIChatTarget()))

# Create the benchmark with scorer and execute with demo dataset
benchmark = QuestionAnsweringBenchmark(
    objective_target=OpenAIChatTarget(),
    attack_scoring_config=scoring_config,
)

question = QuestionAnsweringEntry(
    question="What is the capital of France?",
    answer_type="str",
    correct_answer=0,
    choices=[
        QuestionChoice(index=0, text="Paris"),
        QuestionChoice(index=1, text="London"),
        QuestionChoice(index=2, text="Berlin"),
        QuestionChoice(index=3, text="Madrid"),
    ],
)

result = await benchmark.execute_async(question_answering_entry=question)  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result)  # type: ignore

# %%NBQA-CELL-SEP52c935
# Fetch WMDP dataset for Q/A Model Testing; this is big and can take a minute

wmdp_ds = fetch_wmdp_dataset(category="cyber")

# %%NBQA-CELL-SEP52c935
# Evaluate the Q/A Model response
results = []
for question in wmdp_ds.questions[:3]:
    result = await benchmark.execute_async(question_answering_entry=question)  # type: ignore
    results.append(result)
    await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %%NBQA-CELL-SEP52c935
from pyrit.models import AttackOutcome

success = sum(r.outcome == AttackOutcome.SUCCESS for r in results)
print(f"Success rate: {(success / len(results)) * 100:.2f}%")
