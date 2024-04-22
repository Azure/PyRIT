# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

from datetime import datetime
from unittest.mock import MagicMock
from pyrit.models import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter
from tests.mocks import MockPromptTarget


def test_id_set():
    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello",
        converted_prompt_text="Hello",
    )
    assert entry.id is not None


def test_datetime_set():
    now = datetime.utcnow()
    time.sleep(0.1)
    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello",
        converted_prompt_text="Hello",
    )
    assert entry.timestamp > now


def test_is_sequence_set_false():
    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello",
        converted_prompt_text="Hello",
    )
    assert entry.is_sequence_set() is False


def test_is_sequence_set_true():
    entry = PromptRequestPiece(role="user", original_prompt_text="Hello", converted_prompt_text="Hello", sequence=1)
    assert entry.is_sequence_set()


def test_converters_serialize():
    converter_identifiers = [Base64Converter().get_identifier()]
    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello",
        converted_prompt_text="Hello",
        converter_identifiers=converter_identifiers,
    )

    assert len(entry.converter_identifiers) == 1

    converter = entry.converter_identifiers[0]

    assert converter["__type__"] == "Base64Converter"
    assert converter["__module__"] == "pyrit.prompt_converter.base64_converter"


def test_prompt_targets_serialize():
    target = MockPromptTarget()
    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello",
        converted_prompt_text="Hello",
        prompt_target_identifier=target.get_identifier(),
    )

    assert entry.prompt_target_identifier["__type__"] == "MockPromptTarget"
    assert entry.prompt_target_identifier["__module__"] == "tests.mocks"


def test_orchestrators_serialize():
    orchestrator = PromptSendingOrchestrator(prompt_target=MagicMock(), memory=MagicMock())

    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello",
        converted_prompt_text="Hello",
        orchestrator_identifier=orchestrator.get_identifier(),
    )

    assert entry.orchestrator_identifier["id"] is not None
    assert entry.orchestrator_identifier["__type__"] == "PromptSendingOrchestrator"
    assert entry.orchestrator_identifier["__module__"] == "pyrit.orchestrator.prompt_sending_orchestrator"


def test_hashes_generated():
    entry = PromptRequestPiece(
        role="user",
        original_prompt_text="Hello1",
        converted_prompt_text="Hello2",
    )

    assert entry.original_prompt_data_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"
    assert entry.converted_prompt_data_sha256 == "be98c2510e417405647facb89399582fc499c3de4452b3014857f92e6baad9a9"
