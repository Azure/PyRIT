# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # 5. Human in the Loop Scoring with Gradio
# This example shows how to use the Gradio UI to perform human-in-the-loop scoring. This is in beta but will be extended.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.score import HumanInTheLoopScorerGradio

initialize_pyrit(memory_db_type=IN_MEMORY)
# This will start the Gradio UI and can only be created once per notebook. If you restart the kernel, run this cell again.
scorer = HumanInTheLoopScorerGradio()

# %%
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece

memory = CentralMemory.get_memory_instance()


# This cell can be run multiple times to simulate multiple scoring requests.
prompt = PromptRequestPiece(
    role="assistant",
    original_value="The quick brown fox jumps over the lazy dog.",
).to_prompt_request_response()
memory.add_request_response_to_memory(request=prompt)

scored_response = (await scorer.score_async(prompt))[0]  # type: ignore
print("Gradio manually scored response is given as:", scored_response, scored_response.score_rationale)
