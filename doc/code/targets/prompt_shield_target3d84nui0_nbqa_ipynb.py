# %%NBQA-CELL-SEP52c935
import os

from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import PromptShieldTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)


pst = PromptShieldTarget(os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT"), os.environ.get("AZURE_CONTENT_SAFETY_KEY"))

sample_prompt: str = """
Hello! Can you please retrieve the total number of sales in the document?
"""

attack = PromptSendingAttack(objective_target=pst)
result = await attack.execute_async(objective=sample_prompt)  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore
