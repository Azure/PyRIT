# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Chat service for managing conversations and message handling
"""

import uuid
from datetime import datetime
from typing import List, Optional
import logging

from pyrit.backend.models.responses import ChatResponse, ConversationHistory, Message
from pyrit.backend.services.target_registry import TargetRegistry
from pyrit.models import Message as PyRITMessage, MessagePiece

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations"""

    def __init__(self):
        # In-memory storage for demo purposes
        # TODO: Integrate with PyRIT's memory system
        self.conversations: dict[str, ConversationHistory] = {}

    async def send_message(
        self, message: str, conversation_id: Optional[str] = None, target_id: Optional[str] = None,
        attachments: Optional[list] = None
    ) -> ChatResponse:
        """
        Send a message and get a response

        Args:
            message: The user's message
            conversation_id: Optional conversation ID to continue
            target_id: Optional target ID to use

        Returns:
            ChatResponse with the assistant's reply
        """
        # Create or get conversation
        if conversation_id is None:
            conversation_id = f"conv-{uuid.uuid4()}"

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationHistory(
                conversation_id=conversation_id,
                messages=[],
                target_id=target_id,
            )

        # Add user message
        user_message = Message(role="user", content=message)
        self.conversations[conversation_id].messages.append(user_message)

        # Get response from PyRIT target
        try:
            assistant_response = await self._get_target_response(message, target_id, conversation_id, attachments)
        except Exception as e:
            logger.error(f"Error getting target response: {e}")
            assistant_response = f"Error: {str(e)}"

        # Add assistant message
        assistant_message = Message(role="assistant", content=assistant_response)
        self.conversations[conversation_id].messages.append(assistant_message)
        self.conversations[conversation_id].updated_at = datetime.utcnow()

        return ChatResponse(
            conversation_id=conversation_id,
            message=assistant_response,
            role="assistant",
            target_id=target_id,
        )

    async def get_conversations(self) -> List[ConversationHistory]:
        """Get all conversations"""
        return list(self.conversations.values())

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get a specific conversation by ID"""
        return self.conversations.get(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    async def _get_target_response(self, message: str, target_id: Optional[str], conversation_id: str, attachments: Optional[list] = None) -> str:
        """
        Get response from PyRIT target with multimodal support
        """
        try:
            logger.info(f"Getting target response for conversation {conversation_id}")
            logger.info(f"Message: {message[:100] if message else 'None'}...")
            logger.info(f"Attachments: {len(attachments) if attachments else 0}")
            
            # Create target instance
            if target_id:
                target = TargetRegistry.create_target_instance(target_id)
            else:
                # Use default target
                target = TargetRegistry.get_default_attack_target()
            
            if not target:
                return "Error: Target not configured. Please check your environment variables."
            
            logger.info(f"Using target: {type(target).__name__}")
            
            # Create PyRIT message pieces
            message_pieces = []
            
            # Add text message if present
            if message:
                message_pieces.append(MessagePiece(
                    role="user",
                    conversation_id=conversation_id,
                    original_value=message,
                    converted_value=message,
                    original_value_data_type="text",
                    converted_value_data_type="text",
                ))
            
            # Add attachments as separate message pieces
            if attachments:
                for att in attachments:
                    logger.info(f"Adding attachment: {att['name']} as {att['data_type']}")
                    message_pieces.append(MessagePiece(
                        role="user",
                        conversation_id=conversation_id,
                        original_value=att['path'],
                        converted_value=att['path'],
                        original_value_data_type=att['data_type'],
                        converted_value_data_type=att['data_type'],
                    ))
            
            pyrit_message = PyRITMessage(message_pieces=message_pieces)
            
            logger.info(f"Sending message with {len(message_pieces)} pieces to target")
            # Send to target
            response = await target.send_prompt_async(message=pyrit_message)
            logger.info(f"Received response from target")
            
            # Extract response text
            if response and response.message_pieces:
                return response.message_pieces[0].converted_value
            
            return "No response from target"
            
        except ValueError as e:
            return f"Configuration error: {str(e)}"
        except Exception as e:
            logger.exception("Error in target response")
            return f"Error communicating with target: {str(e)}"
