"""
# TODO: add notebook example instead of this
"""

import asyncio
import logging

from pyrit.executor.attack import ConsoleAttackResultPrinter, MultiPromptSendingAttack
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import WebSocketCopilotTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def main():
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    target = WebSocketCopilotTarget()

    prompts = [
        "I'm thinking of a number between 1 and 10.",
        "It's greater than 5.",
        "It's an even number.",
        "What number am I thinking of?",
    ]

    messages = []
    for prompt in prompts:
        message_piece = MessagePiece(
            role="user",
            original_value=prompt,
            original_value_data_type="text",
            converted_value_data_type="text",
        )
        messages.append(Message(message_pieces=[message_piece]))

    attack = MultiPromptSendingAttack(
        objective_target=target,
    )

    result = await attack.execute_async(
        objective="Engage in a multi-turn conversation about a number guessing game",
        messages=messages,
    )

    printer = ConsoleAttackResultPrinter()
    await printer.print_conversation_async(result=result)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
