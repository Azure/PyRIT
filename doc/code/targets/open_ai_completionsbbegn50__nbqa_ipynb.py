# %%NBQA-CELL-SEP52c935
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAICompletionTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# Note that max_tokens will default to 16 for completions, so you may want to set the upper limit of allowed tokens for a longer response.
target = OpenAICompletionTarget(max_tokens=2048)

attack = PromptSendingAttack(objective_target=target)
result = await attack.execute_async(objective="Hello! Who are you?")  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore
