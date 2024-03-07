import pytest
import textwrap
import json
from pyrit.models import ChatMessage, ChatMessages, ChatMessagesDataset
from dataclasses import asdict


@pytest.fixture
def simple_chat_message() -> ChatMessage:
    message = ChatMessage(
        role="user",
        content="Hello, world!",
    )
    return message


@pytest.fixture
def two_chat_messages() -> list[ChatMessages]:
    messages: list[ChatMessage] = [
        ChatMessage(role="user", content="Hello, world!"),
        ChatMessage(role="assistant", content="Hi, there!", name="bot001"),
    ]
    return messages


def test_single_chat_message_to_chatml(simple_chat_message: ChatMessage) -> None:
    expected_message = textwrap.dedent(
        """\
        <|im_start|>user
        Hello, world!<|im_end|>
        """
    )

    chat_messages_obj = ChatMessages(messages=[simple_chat_message])

    assert chat_messages_obj.to_chatml() == expected_message


def test_two_chat_messages_serialize(two_chat_messages: list[ChatMessages]) -> None:
    expected_message = textwrap.dedent(
        """\
        <|im_start|>user
        Hello, world!<|im_end|>
        <|im_start|>assistant name=bot001
        Hi, there!<|im_end|>
        """
    )

    chat_messages_obj = ChatMessages(messages=two_chat_messages)

    assert chat_messages_obj.to_chatml() == expected_message


def test_chatml_to_messages_when_empty() -> None:
    chatml = textwrap.dedent("empty")

    with pytest.raises(ValueError):
        ChatMessages.from_chatml(chatml)


def test_chatml_to_single_chat_messages() -> None:
    chatml = textwrap.dedent(
        """\
        <|im_start|>user
        Hello, world!<|im_end|>
        """
    )
    expected_messages = [
        ChatMessage(role="user", content="Hello, world!"),
    ]

    chat_messages_obj = ChatMessages.from_chatml(chatml)

    assert chat_messages_obj.messages == expected_messages


def test_chatml_to_chat_messages() -> None:
    chatml = textwrap.dedent(
        """\
        <|im_start|>user
        Hello, world!<|im_end|>
        <|im_start|>assistant name=bot001
        Hi, there!<|im_end|>
        """
    )
    expected_messages = [
        ChatMessage(role="user", content="Hello, world!"),
        ChatMessage(role="assistant", content="Hi, there!", name="bot001"),
    ]

    chat_messages_obj = ChatMessages.from_chatml(chatml)

    assert chat_messages_obj.messages == expected_messages


def test_chat_messages_dataset_values_properly_set() -> None:
    dataset = ChatMessagesDataset(
        description="A dataset of chat messages",
        list_of_chat_messages=[
            ChatMessages(
                messages=[
                    ChatMessage(role="user", content="Hello, world!"),
                    ChatMessage(role="assistant", content="Hi, there!", name="bot001"),
                ]
            ),
            ChatMessages(
                messages=[
                    ChatMessage(role="system", content="you are a helpful AI"),
                    ChatMessage(role="user", content="how are you?"),
                ]
            ),
        ],
        name="test_dataset_001",
    )

    assert dataset.name == "test_dataset_001"
    assert dataset.description == "A dataset of chat messages"
    assert len(dataset.list_of_chat_messages) == 2
    assert len(dataset.list_of_chat_messages[0].messages) == 2
    assert len(dataset.list_of_chat_messages[1].messages) == 2


def test_dataset_object_creation_from_dict() -> None:
    data = {
        "name": "demo-dataset",
        "description": "dataset for demo purposes",
        "list_of_chat_messages": [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant"},
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thanks for asking."},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "Good, you?"},
                ]
            },
        ],
    }

    my_dataset = ChatMessagesDataset(**data)
    assert my_dataset.name == "demo-dataset"
    assert my_dataset.description == "dataset for demo purposes"
    assert len(my_dataset.list_of_chat_messages) == 2
    assert len(my_dataset.list_of_chat_messages[0].messages) == 3
    assert len(my_dataset.list_of_chat_messages[1].messages) == 3
