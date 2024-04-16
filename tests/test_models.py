from pyrit.models import ChatMessage, ChatMessagesDataset


def test_chat_messages_dataset_values_properly_set() -> None:
    dataset = ChatMessagesDataset(
        description="A dataset of chat messages",
        list_of_chat_messages=[
            [
                ChatMessage(role="user", content="Hello, world!"),
                ChatMessage(role="assistant", content="Hi, there!", name="bot001"),
            ],
            [
                ChatMessage(role="system", content="you are a helpful AI"),
                ChatMessage(role="user", content="how are you?"),
            ],
        ],
        name="test_dataset_001",
    )

    assert dataset.name == "test_dataset_001"
    assert dataset.description == "A dataset of chat messages"
    assert len(dataset.list_of_chat_messages) == 2
    assert len(dataset.list_of_chat_messages[0]) == 2
    assert len(dataset.list_of_chat_messages[1]) == 2


def test_dataset_object_creation_from_dict() -> None:
    data = {
        "name": "demo-dataset",
        "description": "dataset for demo purposes",
        "list_of_chat_messages": [
            [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks for asking."},
            ],
            [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good, you?"},
            ],
        ],
    }

    my_dataset = ChatMessagesDataset.model_validate(data)
    assert my_dataset.name == "demo-dataset"
    assert my_dataset.description == "dataset for demo purposes"
    assert len(my_dataset.list_of_chat_messages) == 2
    assert len(my_dataset.list_of_chat_messages[0]) == 3
    assert len(my_dataset.list_of_chat_messages[1]) == 3
