# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from pyrit.message_normalizer.message_normalizer import MessageListNormalizer
from pyrit.models import Message


class MessageNop(MessageListNormalizer[Message]):
    """
    A no-op message normalizer that returns messages unchanged.

    This normalizer is useful when no transformation is needed but a normalizer
    interface is required.
    """

    async def normalize_async(self, messages: List[Message]) -> List[Message]:
        """
        Return the messages unchanged.

        Args:
            messages: The list of messages to normalize.

        Returns:
            The same list of Messages.

        Raises:
            ValueError: If the messages list is empty.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        return list(messages)
