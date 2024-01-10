# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from unittest.mock import Mock, patch

import pytest
import requests

from pyrit.chat.aml_online_endpoint_chat import AMLOnlineEndpointChat
from pyrit.common.net import HttpClientSession
from pyrit.models import ChatMessage

from .mocks import MockHttpPostAsync, MockHttpPostSync

_loop = asyncio.get_event_loop()


@pytest.fixture
def aml_online_chat() -> AMLOnlineEndpointChat:
    aml_online_chat = AMLOnlineEndpointChat(
        endpoint_uri="http://aml-test-endpoint.com",
        api_key="valid_api_key",
    )
    return aml_online_chat


def test_initialization_with_required_parameters(
    aml_online_chat: AMLOnlineEndpointChat,
):
    assert aml_online_chat.endpoint_uri == "http://aml-test-endpoint.com"
    assert aml_online_chat.api_key == "valid_api_key"


def test_get_headers_with_valid_api_key(aml_online_chat: AMLOnlineEndpointChat):
    expected_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer valid_api_key",
    }
    assert aml_online_chat.get_headers() == expected_headers


def test_get_headers_with_empty_api_key(aml_online_chat: AMLOnlineEndpointChat):
    aml_online_chat.api_key = ""
    with pytest.raises(ValueError):
        aml_online_chat.get_headers()


def test_extract_first_response_message_normal(aml_online_chat: AMLOnlineEndpointChat):
    response_message = [{"0": "response from model"}]
    assert (
        aml_online_chat.extract_first_response_message(response_message)
        == "response from model"
    )


def test_extract_first_response_message_empty_list(
    aml_online_chat: AMLOnlineEndpointChat,
):
    with pytest.raises(ValueError) as excinfo:
        aml_online_chat.extract_first_response_message([])
    assert "The response_message list is empty." in str(excinfo.value)


def test_extract_first_response_message_missing_key(
    aml_online_chat: AMLOnlineEndpointChat,
):
    response_message = [{"1": "response from model"}]
    with pytest.raises(ValueError) as excinfo:
        aml_online_chat.extract_first_response_message(response_message)
    assert "Key '0' does not exist in the first response message." in str(excinfo.value)


@patch.object(
    HttpClientSession.get_client_session(), "post", side_effect=MockHttpPostAsync
)
def test_complete_chat_async(
    mock_http_post: Mock, aml_online_chat: AMLOnlineEndpointChat
):
    messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content"),
    ]
    response = _loop.run_until_complete(aml_online_chat.complete_chat_async(messages))
    assert response == "extracted response"
    mock_http_post.assert_called_once()


@patch.object(requests, "post", side_effect=MockHttpPostSync)
def test_complete_chat(mock_http_post: Mock, aml_online_chat: AMLOnlineEndpointChat):
    messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content"),
    ]
    response = aml_online_chat.complete_chat(messages)
    assert response == "extracted response"
    mock_http_post.assert_called_once()
