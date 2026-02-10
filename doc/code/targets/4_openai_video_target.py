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
# `OpenAIVideoTarget` supports three modes:
# - **Text-to-video**: Generate a video from a text prompt.
# - **Remix**: Create a variation of an existing video (using `video_id` from a prior generation).
# - **Image-to-video**: Use an image as the first frame of the generated video.
#
# Note that the video scorer requires `opencv`, which is not a default PyRIT dependency. You need to install it manually or using `pip install pyrit[opencv]`.

# %% [markdown]
# ## Text-to-Video
#
# This example shows the simplest mode: generating video from text prompts, with scoring.

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

# %% [markdown]
# ## Remix (Video Variation)
#
# Remix creates a variation of an existing video. After any successful generation, the response
# includes a `video_id` in `prompt_metadata`. Pass this back via `prompt_metadata={"video_id": "<id>"}` to remix.

# %%
from pyrit.models import Message, MessagePiece

# Use the same target from above, or create a new one
remix_target = OpenAIVideoTarget()

# Step 1: Generate a video
text_piece = MessagePiece(
    role="user",
    original_value="A bird flying over a lake at sunset",
)
result = await remix_target.send_prompt_async(message=Message([text_piece]))  # type: ignore
response = result[0].message_pieces[0]
print(f"Generated video: {response.converted_value}")
video_id = response.prompt_metadata["video_id"]
print(f"Video ID for remix: {video_id}")

# Step 2: Remix using the video_id
remix_piece = MessagePiece(
    role="user",
    original_value="Make it a watercolor painting style",
    prompt_metadata={"video_id": video_id},
)
remix_result = await remix_target.send_prompt_async(message=Message([remix_piece]))  # type: ignore
print(f"Remixed video: {remix_result[0].message_pieces[0].converted_value}")

# %% [markdown]
# ## Image-to-Video
#
# Use an image as the first frame of the generated video. The input image dimensions must match
# the video resolution (e.g. 1280x720). Pass both a text piece and an `image_path` piece in the same message.

# %%
import uuid

# Create a simple test image matching the video resolution (1280x720)
from PIL import Image

from pyrit.common.path import HOME_PATH

sample_image = HOME_PATH / "assets" / "pyrit_architecture.png"
resized = Image.open(sample_image).resize((1280, 720)).convert("RGB")

import tempfile

tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
resized.save(tmp, format="JPEG")
tmp.close()
image_path = tmp.name

# Send text + image to the video target
i2v_target = OpenAIVideoTarget()
conversation_id = str(uuid.uuid4())

text_piece = MessagePiece(
    role="user",
    original_value="Animate this image with gentle camera motion",
    conversation_id=conversation_id,
)
image_piece = MessagePiece(
    role="user",
    original_value=image_path,
    converted_value_data_type="image_path",
    conversation_id=conversation_id,
)
result = await i2v_target.send_prompt_async(message=Message([text_piece, image_piece]))  # type: ignore
print(f"Image-to-video result: {result[0].message_pieces[0].converted_value}")
