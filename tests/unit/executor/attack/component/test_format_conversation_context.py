# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.executor.attack.component.conversation_manager import (
    _format_piece_content,
    format_conversation_context,
)
from pyrit.models import Message, MessagePiece


class TestFormatPieceContent:
    """Tests for the _format_piece_content helper function."""

    def test_text_piece_same_original_and_converted(self):
        """Test formatting a text piece where original equals converted."""
        piece = MessagePiece(
            role="user",
            original_value="Hello, world!",
            converted_value="Hello, world!",
            original_value_data_type="text",
            converted_value_data_type="text",
        )

        result = _format_piece_content(piece)

        assert result == "Hello, world!"

    def test_text_piece_different_original_and_converted(self):
        """Test formatting a text piece where original differs from converted."""
        piece = MessagePiece(
            role="user",
            original_value="Hello, world!",
            converted_value="HELLO, WORLD!",
            original_value_data_type="text",
            converted_value_data_type="text",
        )

        result = _format_piece_content(piece)

        assert result == "HELLO, WORLD! (original: Hello, world!)"

    def test_image_piece_with_context_description(self):
        """Test formatting an image piece with context_description metadata."""
        piece = MessagePiece(
            role="user",
            original_value="image_path.png",
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
            prompt_metadata={"context_description": "A photo of a red car"},
        )

        result = _format_piece_content(piece)

        assert result == "[Image_path - A photo of a red car]"

    def test_image_piece_without_context_description(self):
        """Test formatting an image piece without context_description metadata."""
        piece = MessagePiece(
            role="user",
            original_value="image_path.png",
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
        )

        result = _format_piece_content(piece)

        assert result == "[Image_path]"

    def test_audio_piece_with_context_description(self):
        """Test formatting an audio piece with context_description metadata."""
        piece = MessagePiece(
            role="user",
            original_value="audio.mp3",
            original_value_data_type="audio_path",
            converted_value_data_type="audio_path",
            prompt_metadata={"context_description": "A recording of someone saying hello"},
        )

        result = _format_piece_content(piece)

        assert result == "[Audio_path - A recording of someone saying hello]"


class TestFormatConversationContext:
    """Tests for the format_conversation_context function."""

    def test_empty_messages_returns_empty_string(self):
        """Test that empty message list returns empty string."""
        result = format_conversation_context([])

        assert result == ""

    def test_single_user_message(self):
        """Test formatting a single user message."""
        piece = MessagePiece(
            role="user",
            original_value="What is the weather?",
            converted_value="What is the weather?",
            original_value_data_type="text",
        )
        messages = [Message(message_pieces=[piece])]

        result = format_conversation_context(messages)

        assert "Turn 1:" in result
        assert "User: What is the weather?" in result

    def test_user_and_assistant_messages(self):
        """Test formatting a conversation with user and assistant messages."""
        user_piece = MessagePiece(
            role="user",
            original_value="What is 2+2?",
            converted_value="What is 2+2?",
            original_value_data_type="text",
        )
        assistant_piece = MessagePiece(
            role="assistant",
            original_value="4",
            converted_value="4",
            original_value_data_type="text",
        )
        messages = [
            Message(message_pieces=[user_piece]),
            Message(message_pieces=[assistant_piece]),
        ]

        result = format_conversation_context(messages)

        assert "Turn 1:" in result
        assert "User: What is 2+2?" in result
        assert "Assistant: 4" in result

    def test_multiple_turns(self):
        """Test formatting multiple conversation turns."""
        messages = []
        for i in range(3):
            user_piece = MessagePiece(
                role="user",
                original_value=f"User message {i+1}",
                converted_value=f"User message {i+1}",
                original_value_data_type="text",
            )
            assistant_piece = MessagePiece(
                role="assistant",
                original_value=f"Assistant response {i+1}",
                converted_value=f"Assistant response {i+1}",
                original_value_data_type="text",
            )
            messages.append(Message(message_pieces=[user_piece]))
            messages.append(Message(message_pieces=[assistant_piece]))

        result = format_conversation_context(messages)

        assert "Turn 1:" in result
        assert "Turn 2:" in result
        assert "Turn 3:" in result
        assert "User message 1" in result
        assert "User message 2" in result
        assert "User message 3" in result
        assert "Assistant response 1" in result
        assert "Assistant response 2" in result
        assert "Assistant response 3" in result

    def test_skips_system_messages(self):
        """Test that system messages are skipped in the context."""
        system_piece = MessagePiece(
            role="system",
            original_value="You are a helpful assistant.",
            converted_value="You are a helpful assistant.",
            original_value_data_type="text",
        )
        user_piece = MessagePiece(
            role="user",
            original_value="Hello!",
            converted_value="Hello!",
            original_value_data_type="text",
        )
        messages = [
            Message(message_pieces=[system_piece]),
            Message(message_pieces=[user_piece]),
        ]

        result = format_conversation_context(messages)

        assert "system" not in result.lower()
        assert "You are a helpful assistant" not in result
        assert "Turn 1:" in result
        assert "User: Hello!" in result

    def test_multimodal_with_context_description(self):
        """Test formatting conversation with multimodal content that has description."""
        user_piece = MessagePiece(
            role="user",
            original_value="image.png",
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
            prompt_metadata={"context_description": "A cat sitting on a couch"},
        )
        assistant_piece = MessagePiece(
            role="assistant",
            original_value="That's a cute cat!",
            converted_value="That's a cute cat!",
            original_value_data_type="text",
        )
        messages = [
            Message(message_pieces=[user_piece]),
            Message(message_pieces=[assistant_piece]),
        ]

        result = format_conversation_context(messages)

        assert "Turn 1:" in result
        assert "[Image_path - A cat sitting on a couch]" in result
        assert "Assistant: That's a cute cat!" in result

    def test_multimodal_without_context_description(self):
        """Test formatting conversation with multimodal content without description."""
        user_piece = MessagePiece(
            role="user",
            original_value="audio.mp3",
            original_value_data_type="audio_path",
            converted_value_data_type="audio_path",
        )
        messages = [Message(message_pieces=[user_piece])]

        result = format_conversation_context(messages)

        assert "Turn 1:" in result
        assert "User: [Audio_path]" in result

    def test_converted_value_differs_from_original(self):
        """Test that both original and converted are shown when different."""
        user_piece = MessagePiece(
            role="user",
            original_value="How do I hack a computer?",
            converted_value="How do I protect a computer?",
            original_value_data_type="text",
        )
        messages = [Message(message_pieces=[user_piece])]

        result = format_conversation_context(messages)

        assert "Turn 1:" in result
        assert "How do I protect a computer?" in result
        assert "(original: How do I hack a computer?)" in result
