# %%NBQA-CELL-SEP52c935
import os

from pyrit.memory import CentralMemory
from pyrit.models import Message, MessagePiece
from pyrit.score.float_scale.azure_content_filter_scorer import AzureContentFilterScorer
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)


# Set up the Azure Content Filter
azure_content_filter = AzureContentFilterScorer(
    # Comment out either api_key or use_entra_auth
    api_key=os.environ.get("AZURE_CONTENT_SAFETY_API_KEY"),
    # use_entra_auth=True,
    endpoint=os.environ.get("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
)

response = Message(
    message_pieces=[
        MessagePiece(
        role="assistant",
        original_value_data_type="text",
        original_value="I hate you.",
        )
    ]
)
memory = CentralMemory.get_memory_instance()
# need to write it manually to memory as score table has a foreign key constraint
memory.add_message_to_memory(request=response)

# Run the request
scores = await azure_content_filter.score_async(response)  # type: ignore
assert scores[0].get_value() > 0  # azure_severity should be value 2 based on the documentation

for score in scores:
    # score_metadata contains azure_severity original value
    print(f"{score} {score.score_metadata}")
