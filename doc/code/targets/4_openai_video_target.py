# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
# ---

# %% [markdown]
# # 4. OpenAI Video Target
#
# This example shows how to use the video target to create a video from a text prompt.
#
# Note that the video scorer requires `opencv`, which is not a default PyRIT dependency. You need to install it manually or using `pip install pyrit[opencv]`.

# %%
from pyrit.executor.attack import (
    AttackExecutor,
    AttackScoringConfig,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.prompt_target import OpenAIChatTarget, OpenAIVideoTarget
from pyrit.score import (
    AzureContentFilterScorer,
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
    VideoFloatScaleScorer,
    VideoTrueFalseScorer,
)
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

video_target = OpenAIVideoTarget()
objective_scorer = SelfAskTrueFalseScorer(
    chat_target=OpenAIChatTarget(),
    true_false_question=TrueFalseQuestion(true_description="A raccoon dressed as a pirate is actively eating a pastry"),
)

video_scorer = VideoTrueFalseScorer(
    image_capable_scorer=objective_scorer,
    num_sampled_frames=10,
)

attack = PromptSendingAttack(
    objective_target=video_target,
    attack_scoring_config=AttackScoringConfig(
        objective_scorer=video_scorer,
        auxiliary_scorers=[VideoFloatScaleScorer(image_capable_scorer=AzureContentFilterScorer())],
    ),
)

objectives = [
    "Video of a raccoon pirate eating flan at a cafe in Spain",
    "Video of a raccoon pirate eating a croissant at a cafe in France",
]

results = await AttackExecutor().execute_attack_async(  # type: ignore
    attack=attack,
    objectives=objectives,
)

for result in results:
    await ConsoleAttackResultPrinter().print_result_async(result=result, include_auxiliary_scores=True)  # type: ignore
