# %%NBQA-CELL-SEP52c935
from pyrit.memory import CentralMemory
from pyrit.score import HumanInTheLoopScorerGradio
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)
# This will start the Gradio UI and can only be created once per notebook. If you restart the kernel, run this cell again.
scorer = HumanInTheLoopScorerGradio() 

# %%NBQA-CELL-SEP52c935
from pyrit.memory import CentralMemory
from pyrit.models import MessagePiece

memory = CentralMemory.get_memory_instance()


# This cell can be run multiple times to simulate multiple scoring requests.
prompt = MessagePiece(
    role="assistant",
    original_value="The quick brown fox jumps over the lazy dog.",
).to_message()
memory.add_message_to_memory(request=prompt)

scored_response = (await scorer.score_async(prompt))[0]  # type: ignore
print("Gradio manually scored response is given as:", scored_response, scored_response.score_rationale)
