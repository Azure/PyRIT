# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import hashlib
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ConfigDict
from pyrit.models.chat_message import ChatMessage


ALLOWED_CHAT_MESSAGE_ROLES = ["system", "user", "assistant"]


class PromptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The text response for the prompt
    completion: str
    # The original prompt
    prompt: str = ""
    # An unique identifier for the response
    id: str = ""
    # The number of tokens used in the completion
    completion_tokens: int = 0
    # The number of tokens sent in the prompt
    prompt_tokens: int = 0
    # Total number of tokens used in the request
    total_tokens: int = 0
    # The model used
    model: str = ""
    # The type of operation (e.g., "text_completion")
    object: str = ""
    # When the object was created
    created_at: int = 0
    logprobs: Optional[bool] = False
    index: int = 0
    # Rationale why the model ended (e.g., "stop")
    finish_reason: str = ""
    # The time it took to complete the request from the moment the API request
    # was made, in nanoseconds.
    api_request_time_to_complete_ns: int = 0

    # Extra metadata that can be added to the response
    metadata: dict = {}

    def save_to_file(self, directory_path: Path) -> str:
        """Save the Prompt Response to disk and return the path of the new file.

        Args:
            directory_path: The path to save the file to
        Returns:
            The full path to the file that was saved
        """
        embedding_json = self.json()
        embedding_hash = hashlib.sha256(embedding_json.encode()).hexdigest()
        embedding_output_file_path = Path(directory_path, f"{embedding_hash}.json")
        embedding_output_file_path.write_text(embedding_json)
        return embedding_output_file_path.as_posix()

    def to_json(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def load_from_file(file_path: Path) -> PromptResponse:
        """Load the Prompt Response from disk

        Args:
            file_path: The path to load the file from
        Returns:
            The loaded embedding response
        """
        embedding_json_data = file_path.read_text(encoding="utf-8")
        return PromptResponse.model_validate_json(embedding_json_data)


@dataclass
class Prompt:
    content: str


class QuestionChoice(BaseModel):
    """
    Represents a choice for a question.

    Attributes:
        index (int): The index of the choice.
        text (str): The text of the choice.
    """

    model_config = ConfigDict(extra="forbid")
    index: int
    text: str


class QuestionAnsweringEntry(BaseModel):
    """
    Represents a question model.

    Attributes:
        question (str): The question text.
        answer_type (Literal["int", "float", "str", "bool"]): The type of the answer.
            - `int` for integer answers (e.g., when the answer is an index of the correct option in a multiple-choice
               question).
            - `float` for answers that are floating-point numbers.
            - `str` for text-based answers.
            - `bool` for boolean answers.
        correct_answer (Union[int, str, float]): The correct answer.
        choices (list[QuestionChoice]): The list of choices for the question.
    """

    model_config = ConfigDict(extra="forbid")
    question: str
    answer_type: Literal["int", "float", "str", "bool"]
    correct_answer: Union[int, str, float]
    choices: list[QuestionChoice]

    def __hash__(self):
        return hash(self.model_dump_json())


class QuestionAnsweringDataset(BaseModel):
    """
    Represents a dataset for question answering.

    Attributes:
        name (str): The name of the dataset.
        version (str): The version of the dataset.
        description (str): A description of the dataset.
        author (str): The author of the dataset.
        group (str): The group associated with the dataset.
        source (str): The source of the dataset.
        questions (list[QuestionAnsweringEntry]): A list of question models.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = ""
    version: str = ""
    description: str = ""
    author: str = ""
    group: str = ""
    source: str = ""
    questions: list[QuestionAnsweringEntry]


T = TypeVar("T", bound="YamlLoadable")


class YamlLoadable(abc.ABC):
    """
    Abstract base class for objects that can be loaded from YAML files.
    """

    @classmethod
    def from_yaml_file(cls: Type[T], file: Path) -> T:
        """
        Creates a new object from a YAML file.

        Args:
            file: The input file path.

        Returns:
            A new object of type T.

        Raises:
            FileNotFoundError: If the input YAML file path does not exist.
            ValueError: If the YAML file is invalid.
        """
        if not file.exists():
            raise FileNotFoundError(f"File '{file}' does not exist.")
        try:
            yaml_data = yaml.safe_load(file.read_text("utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML file '{file}': {exc}")
        data_object = cls(**yaml_data)
        return data_object


@dataclass
class PromptDataset(YamlLoadable):
    name: str
    description: str
    harm_category: str
    should_be_blocked: bool
    author: str = ""
    group: str = ""
    source: str = ""
    prompts: list[str] = field(default_factory=list)


class ChatMessagesDataset(BaseModel):
    """
    Represents a dataset of chat messages.

    Attributes:
        model_config (ConfigDict): The model configuration.
        name (str): The name of the dataset.
        description (str): The description of the dataset.
        list_of_chat_messages (list[list[ChatMessage]]): A list of chat messages.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    list_of_chat_messages: list[list[ChatMessage]]


@dataclass
class PromptTemplate(YamlLoadable):
    template: str
    name: str = ""
    description: str = ""
    should_be_blocked: bool = False
    harm_category: str = ""
    author: str = ""
    group: str = ""
    source: str = ""
    parameters: list[str] = field(default_factory=list)

    def apply_custom_metaprompt_parameters(self, **kwargs) -> str:
        """Builds a new prompts from the metaprompt template.
        Args:
            **kwargs: the key value for the metaprompt template inputs

        Returns:
            A new prompt following the template
        """
        final_prompt = self.template
        for key, value in kwargs.items():
            if key not in self.parameters:
                raise ValueError(f'Invalid parameters passed. [expected="{self.parameters}", actual="{kwargs}"]')
            # Matches field names within brackets {{ }}
            #  {{   key    }}
            #  ^^^^^^^^^^^^^^
            regex = "{}{}{}".format("\{\{ *", key, " *\}\}")  # noqa: W605
            matches = re.findall(pattern=regex, string=final_prompt)
            if not matches:
                raise ValueError(
                    f"No parameters matched, they might be missing in the template. "
                    f'[expected="{self.parameters}", actual="{kwargs}"]'
                )
            final_prompt = re.sub(pattern=regex, string=final_prompt, repl=value)
        return final_prompt


@dataclass
class AttackStrategy:
    def __init__(self, *, strategy: Union[Path | str], **kwargs):
        self.kwargs = kwargs
        if isinstance(strategy, Path):
            self.strategy = PromptTemplate.from_yaml_file(strategy)
        else:
            self.strategy = PromptTemplate(template=strategy, parameters=list(kwargs.keys()))

    def __str__(self):
        """Returns a string representation of the attack strategy."""
        return self.strategy.apply_custom_metaprompt_parameters(**self.kwargs)


class EmbeddingUsageInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding: list[float]
    index: int
    object: str


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    object: str
    usage: EmbeddingUsageInformation
    data: list[EmbeddingData]

    def save_to_file(self, directory_path: Path) -> str:
        """Save the embedding response to disk and return the path of the new file

        Args:
            directory_path: The path to save the file to
        Returns:
            The full path to the file that was saved
        """
        embedding_json = self.json()
        embedding_hash = sha256(embedding_json.encode()).hexdigest()
        embedding_output_file_path = Path(directory_path, f"{embedding_hash}.json")
        embedding_output_file_path.write_text(embedding_json)
        return embedding_output_file_path.as_posix()

    @staticmethod
    def load_from_file(file_path: Path) -> EmbeddingResponse:
        """Load the embedding response from disk

        Args:
            file_path: The path to load the file from
        Returns:
            The loaded embedding response
        """
        embedding_json_data = file_path.read_text(encoding="utf-8")
        return EmbeddingResponse.model_validate_json(embedding_json_data)

    def to_json(self) -> str:
        return self.model_dump_json()
