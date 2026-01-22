# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # 5.1 OpenAI Chat Target (Text + Image --> Text)
#
# This example shows how to use the chat target to generate a text output from a combination of a text prompt and an image input.

# %%
from pathlib import Path

from pyrit.executor.attack import (
    AttackScoringConfig,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.models import SeedGroup, SeedPrompt
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestion
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore


chat_target = OpenAIChatTarget()

scorer = SelfAskTrueFalseScorer(
    chat_target=OpenAIChatTarget(),
    true_false_question=TrueFalseQuestion(
        true_description="The response describes the picture as an overview of PyRIT components."
    ),
)

scoring_config = AttackScoringConfig(objective_scorer=scorer)

attack = PromptSendingAttack(
    objective_target=chat_target,
    attack_scoring_config=scoring_config,
)

# use the image from our docs
image_path = str(Path(".") / ".." / ".." / ".." / "assets" / "pyrit_architecture.png")

# This is a single request with two parts, one image and one text
seed = SeedGroup(
    seeds=[
        SeedPrompt(
            value="Describe this picture:",
            data_type="text",
        ),
        SeedPrompt(
            value=str(image_path),
            data_type="image_path",
        ),
    ]
)

result = await attack.execute_async(
    objective="Describe the picture",
    next_message=seed.next_message,
)  # type: ignore

await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore
