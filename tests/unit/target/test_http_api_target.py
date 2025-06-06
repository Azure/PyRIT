# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target.http_target.httpx_api_target import HTTPXAPITarget


@pytest.mark.asyncio
@patch("httpx.AsyncClient.request")
async def test_send_prompt_async_file_upload(mock_request, patch_central_database):
    # Create a temporary file to simulate a PDF.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"This is a mock PDF content")
        tmp.flush()
        file_path = tmp.name

    # Create a PromptRequestPiece with converted_value set to the temporary file path.
    request_piece = PromptRequestPiece(role="user", original_value="mock", converted_value=file_path)
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Mock a response simulating a file upload.
    mock_response = MagicMock()
    mock_response.content = b'{"message": "File uploaded successfully", "filename": "mock.pdf"}'
    mock_request.return_value = mock_response

    # Create HTTPXAPITarget without passing a transport.
    target = HTTPXAPITarget(http_url="http://example.com/upload/", method="POST", timeout=180)
    response = await target.send_prompt_async(prompt_request=prompt_request)

    # Our mock transport returns a JSON string containing "File uploaded successfully".
    response_text = (
        str(response.request_pieces[0].converted_value) if response.request_pieces[0].converted_value else str(response)
    )
    assert "File uploaded successfully" in response_text

    # Clean up the temporary file.
    os.unlink(file_path)


@pytest.mark.asyncio
@patch("httpx.AsyncClient.request")
async def test_send_prompt_async_no_file(mock_request, patch_central_database):
    # Create a PromptRequestPiece with converted_value that does not point to a valid file.
    request_piece = PromptRequestPiece(role="user", original_value="mock", converted_value="non_existent_file.pdf")
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Mock a response simulating a standard API (non-file).
    mock_response = MagicMock()
    mock_response.content = b'{"status": "ok", "data": "Sample JSON response"}'
    mock_request.return_value = mock_response

    target = HTTPXAPITarget(http_url="http://example.com/data/", method="POST", timeout=180)
    response = await target.send_prompt_async(prompt_request=prompt_request)

    # The mock transport returns a JSON string containing "Sample JSON response".
    response_text = (
        str(response.request_pieces[0].converted_value) if response.request_pieces[0].converted_value else str(response)
    )
    assert "Sample JSON response" in response_text


@pytest.mark.asyncio
@patch("httpx.AsyncClient.request")
async def test_send_prompt_async_validation(mock_request, patch_central_database):
    # Create an invalid prompt request (empty request_pieces)
    prompt_request = PromptRequestResponse(request_pieces=[])
    target = HTTPXAPITarget(http_url="http://example.com/validate/", method="POST", timeout=180)

    with pytest.raises(ValueError) as excinfo:
        await target.send_prompt_async(prompt_request=prompt_request)

    assert "This target only supports a single prompt request piece." in str(excinfo.value)
