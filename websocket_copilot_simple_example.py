"""
# TODO
THIS WILL BE REMOVED after proper unit tests are in place :)
"""

import asyncio

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import WebSocketCopilotTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async


async def main():
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)
    target = WebSocketCopilotTarget()

    message_piece = MessagePiece(
        role="user",
        original_value="say only one random word",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    message = Message(message_pieces=[message_piece])

    responses = await target.send_prompt_async(message=message)
    for response in responses:
        print(f"{response.get_value()}")


if __name__ == "__main__":
    asyncio.run(main())
