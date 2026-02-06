# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pyrit
#     language: python
#     name: python3
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
    AudioTrueFalseScorer,
    AzureContentFilterScorer,
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
    VideoFloatScaleScorer,
    VideoTrueFalseScorer,
)
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

video_target = OpenAIVideoTarget()

# %% [markdown]
# ## Generating and scoring a video:
#
# Using the video target you can send prompts to generate a video. The video scorer can evaluate the video content itself. Note this section is simply scoring the **video** not the audio.

# %%
objectives = [
    "Video of a raccoon pirate eating flan at a cafe in Spain",
    # "Video of a raccoon pirate eating a croissant at a cafe in France",
]

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

results = await AttackExecutor().execute_attack_async(  # type: ignore
    attack=attack,
    objectives=objectives,
)

for result in results:
    await ConsoleAttackResultPrinter().print_result_async(result=result, include_auxiliary_scores=True)  # type: ignore

# %% [markdown]
# ## Scoring video and audio **together**:
#
# The audio scorer can be added in in order to evaluate both the video content and the audio present in the generated video.
#
# *Note*: the current audio scoring will use transcription, so if the audio is not able to be transcribed this will return False

# %%
# Scorer for audio content (transcript) - checks what is HEARD/SAID in the video
objectives = [
    "Video of a raccoon pirate eating a croissant at a cafe in France who says 'Bonjour!, my name is Roakey and this is the best croissant ever!' Ensure the video contains all the audio.",
    # "Video of a raccoon pirate eating a croissant at a cafe in Spain who says 'Hola a todos, my name is Roakey and I am in Spain!' Ensure the video contains all the audio.",
]

# Visual scorer - checks what is SEEN in the video frames
visual_scorer = SelfAskTrueFalseScorer(
    chat_target=OpenAIChatTarget(),
    true_false_question=TrueFalseQuestion(true_description="A raccoon dressed as a pirate is actively eating a pastry"),
)

# Audio transcript scorer - checks what is SAID in the video
audio_text_scorer = SelfAskTrueFalseScorer(
    chat_target=OpenAIChatTarget(),
    true_false_question=TrueFalseQuestion(
        true_description="Someone introduces themselves and expresses enjoyment of a croissant"
    ),
)

# VideoTrueFalseScorer uses hard-coded aggregation:
# - Frame scores are aggregated with OR (True if ANY frame matches the objective)
# - When audio_scorer is provided, the final score uses AND (BOTH video AND audio must be True)
audio_and_video_scorer = VideoTrueFalseScorer(
    image_capable_scorer=visual_scorer,
    num_sampled_frames=3,
    audio_scorer=AudioTrueFalseScorer(text_capable_scorer=audio_text_scorer),
)

attack = PromptSendingAttack(
    objective_target=video_target,
    attack_scoring_config=AttackScoringConfig(
        objective_scorer=audio_and_video_scorer,
    ),
)

results = await AttackExecutor().execute_attack_async(  # type: ignore
    attack=attack,
    objectives=objectives,
)

for result in results:
    await ConsoleAttackResultPrinter().print_result_async(result=result, include_auxiliary_scores=True)  # type: ignore
