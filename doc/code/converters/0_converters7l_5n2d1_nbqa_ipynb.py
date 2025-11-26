# %%NBQA-CELL-SEP52c935
from pyrit.prompt_converter import (
    AsciiArtConverter,
    BinaryConverter,
    FirstLetterConverter,
    RandomCapitalLettersConverter,
    ROT13Converter,
)
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

prompt = "tell me how to cut down a tree"


print(await ROT13Converter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await RandomCapitalLettersConverter(percentage=25.0).convert_tokens_async(prompt=prompt))  # type: ignore
print(await AsciiArtConverter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await BinaryConverter().convert_tokens_async(prompt=prompt))  # type: ignore
print(await FirstLetterConverter().convert_tokens_async(prompt=prompt))  # type: ignore

# %%NBQA-CELL-SEP52c935
# Close connection
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
