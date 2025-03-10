# %% [markdown]
# # Human in the Loop Scoring with Gradio
# This example shows how to use the Gradio UI to perform human-in-the-loop scoring.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece
from pyrit.score import HumanInTheLoopScorerGradio

initialize_pyrit(memory_db_type=IN_MEMORY)
memory = CentralMemory.get_memory_instance()
# This will start the Gradio UI and can only be created once per notebook. If you restart the kernel, run this cell again.
scorer = HumanInTheLoopScorerGradio()

# %%
# This cell can be run multiple times to simulate multiple scoring requests.
prompt = PromptRequestPiece(
    role="assistant",
    original_value="The quick brown fox jumps over the lazy dog.",
)
memory.add_request_pieces_to_memory(request_pieces=[prompt])

scored_response = (await scorer.score_async(prompt))[0]  # type: ignore
print("Gradio manually scored response is given as:", scored_response, scored_response.score_rationale)

# %%
scorer.__del__()
memory.dispose_engine()
