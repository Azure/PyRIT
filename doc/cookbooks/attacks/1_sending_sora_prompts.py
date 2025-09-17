import asyncio
import os

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_target import OpenAISoraTarget
from pyrit.executor.attack import (
    AttackExecutor,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)

def initialize():
    initialize_pyrit(memory_db_type="InMemory")

async def test():
    initialize()
    sora_target = get_objective_target()

    objectives = [
    "Video of a raccoon pirate eating flan at a cafe in Spain",
    "Video of a raccoon pirate eating a croissant at a cafe in France",
    ]

    attack = PromptSendingAttack(objective_target=sora_target)

    results = await AttackExecutor().execute_multi_objective_attack_async(  # type: ignore
    attack=attack,
    objectives=objectives,
    )

    for result in results:
        await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore


def get_objective_target():
    return OpenAISoraTarget(endpoint=os.environ["AZURE_OPENAI_SORA_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_SORA_KEY"])
    # return OpenAISoraTarget(use_aad_auth=True)

async def main():
    await test()

asyncio.run(main())