# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import patch

import pytest

from pyrit.prompt_target import WebSocketCopilotTarget


VALID_WEBSOCKET_URL = (
    "wss://substrate.office.com/m365Copilot/Chathub/test_object_id@test_tenant_id"
    "?ClientRequestId=test_client_request_id"
    "&X-SessionId=test_session_id&token=abc123"
    "&ConversationId=test_conversation_id"
    "&access_token=test_access_token"
    # "&variants=feature.test_feature_one,feature.test_feature_two"
    # "&agent=web"
    # "&scenario=OfficeWebIncludedCopilot"
)


@pytest.fixture
def mock_env_websocket_url():
    with patch.dict(os.environ, {"WEBSOCKET_URL": VALID_WEBSOCKET_URL}):
        yield


@pytest.mark.usefixtures("patch_central_database")
class TestWebSocketCopilotTargetInit:
    def test_init_with_valid_wss_url(self, mock_env_websocket_url):
        target = WebSocketCopilotTarget()

        assert target._websocket_url == VALID_WEBSOCKET_URL
        assert target._conversation_id == "test_conversation_id"
        assert target._session_id == "test_session_id"
        assert target._model_name == "copilot"

    def test_init_with_missing_or_invalid_wss_url(self):
        for env_vars in [{}, {"WEBSOCKET_URL": ""}, {"WEBSOCKET_URL": "   "}]:
            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(ValueError, match="WebSocket URL must be provided"):
                    WebSocketCopilotTarget()

        for invalid_url in ["invalid_websocket_url", "ws://example.com", "https://example.com"]:
            with patch.dict(os.environ, {"WEBSOCKET_URL": invalid_url}, clear=True):
                with pytest.raises(ValueError, match="WebSocket URL must start with 'wss://'"):
                    WebSocketCopilotTarget()

    def test_init_with_missing_or_empty_required_params(self):
        urls = [
            ("wss://example.com/?X-SessionId=session123", "`ConversationId` parameter not found"),
            ("wss://example.com/?ConversationId=conv123", "`X-SessionId` parameter not found"),
            ("wss://example.com/?ConversationId=&X-SessionId=session123", "`ConversationId` parameter is empty"),
            ("wss://example.com/?ConversationId=conv123&X-SessionId=", "`X-SessionId` parameter is empty"),
        ]

        for url, error_msg in urls:
            with patch.dict(os.environ, {"WEBSOCKET_URL": url}, clear=True):
                with pytest.raises(ValueError, match=error_msg):
                    WebSocketCopilotTarget()

    def test_init_sets_endpoint_correctly(self, mock_env_websocket_url):
        target = WebSocketCopilotTarget()
        assert target._endpoint == "wss://substrate.office.com/m365Copilot/Chathub/test_object_id@test_tenant_id"

    def test_init_with_custom_response_timeout(self, mock_env_websocket_url):
        target = WebSocketCopilotTarget(response_timeout_seconds=120)
        assert target._response_timeout_seconds == 120

        for invalid_timeout in [0, -10]:
            with pytest.raises(ValueError, match="response_timeout_seconds must be a positive integer."):
                WebSocketCopilotTarget(response_timeout_seconds=invalid_timeout)


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"key": "value"}, '{"key":"value"}\x1e'),
        ({"protocol": "json", "version": 1}, '{"protocol":"json","version":1}\x1e'),
        ({"outer": {"inner": "value"}}, '{"outer":{"inner":"value"}}\x1e'),
        ({"items": [1, 2, 3]}, '{"items":[1,2,3]}\x1e'),
    ],
)
def test_dict_to_websocket_static_method(data, expected):
    result = WebSocketCopilotTarget._dict_to_websocket(data)
    assert result == expected


class TestParseRawMessage:
    from pyrit.prompt_target.websocket_copilot_target import CopilotMessageType

    @pytest.mark.parametrize(
        "message,expected_types,expected_content",
        [
            ("", [CopilotMessageType.UNKNOWN], [""]),
            ("   \n\t  ", [CopilotMessageType.UNKNOWN], [""]),
            ("{}\x1e", [CopilotMessageType.UNKNOWN], [""]),
            ('{"type":6}\x1e', [CopilotMessageType.PING], [""]),
            (
                '{"type":1,"target":"update","arguments":[{"messages":[{"text":"Partial","author":"bot"}]}]}\x1e',
                [CopilotMessageType.PARTIAL_RESPONSE],
                [""],
            ),
            (
                '{"type":2,"item":{"result":{"message":"Final."}}}\x1e{"type":3,"invocationId":"0"}\x1e',
                [CopilotMessageType.FINAL_CONTENT, CopilotMessageType.STREAM_END],
                [
                    "Final.",
                    "",
                ],
            ),
        ],
    )
    def test_parse_raw_message_with_valid_data(self, message, expected_types, expected_content):
        result = WebSocketCopilotTarget._parse_raw_message(message)

        assert len(result) == len(expected_types)
        for i, expected_type in enumerate(expected_types):
            assert result[i][0] == expected_type
            assert result[i][1] == expected_content[i]

    def test_parse_final_message_without_content(self):
        from pyrit.prompt_target.websocket_copilot_target import CopilotMessageType

        with patch("pyrit.prompt_target.websocket_copilot_target.logger") as mock_logger:
            message = '{"type":2,"invocationId":"0"}\x1e'
            result = WebSocketCopilotTarget._parse_raw_message(message)

            assert len(result) == 1
            assert result[0][0] == CopilotMessageType.FINAL_CONTENT
            assert result[0][1] == ""

            mock_logger.warning.assert_called_with("FINAL_CONTENT received but no parseable content found.")
            mock_logger.debug.assert_called_with(f"Full raw message: {message[:-1]}")

    @pytest.mark.parametrize(
        "message",
        [
            '{"type":99,"data":"unknown"}\x1e',
            '{"data":"no type field"}\x1e',
            '{"invalid json structure\x1e',
        ],
    )
    def test_parse_unknown_or_invalid_messages(self, message):
        from pyrit.prompt_target.websocket_copilot_target import CopilotMessageType

        result = WebSocketCopilotTarget._parse_raw_message(message)
        assert len(result) == 1
        assert result[0][0] == CopilotMessageType.UNKNOWN
        assert result[0][1] == ""
