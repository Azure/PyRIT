# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile

import httpx
import pytest

from pyrit.common import DUCK_DB, initialize_pyrit
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target.http_target.httpx_api_target import HTTPXApiTarget

# Initialize PyRIT to set up the central memory instance.
initialize_pyrit(memory_db_type=DUCK_DB)


# Dummy transport to simulate a successful file upload response.
def dummy_transport_file(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b'{"message": "File uploaded successfully", "filename": "dummy.pdf"}')


# Dummy transport to simulate a standard API (non-file) response.
def dummy_transport_no_file(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b'{"status": "ok", "data": "Sample JSON response"}')


@pytest.mark.asyncio
async def test_send_prompt_async_file_upload():
    # Create a temporary file to simulate a PDF.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"This is a dummy PDF content")
        tmp.flush()
        file_path = tmp.name

    # Create a PromptRequestPiece with converted_value set to the temporary file path.
    request_piece = PromptRequestPiece(role="user", original_value="dummy", converted_value=file_path)
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Create HTTPXApiTarget with a dummy transport that simulates a file upload response.
    transport = httpx.MockTransport(dummy_transport_file)
    target = HTTPXApiTarget(
        http_url="http://example.com/upload/", method="POST", timeout=180, **{"transport": transport}
    )

    # Call the async method; pytest-asyncio handles the event loop internally.
    response = await target.send_prompt_async(prompt_request=prompt_request)

    # Our dummy transport returns a JSON string containing "File uploaded successfully".
    response_text = (
        str(response.request_pieces[0].converted_value) if response.request_pieces[0].converted_value else str(response)
    )
    assert "File uploaded successfully" in response_text

    # Clean up the temporary file.
    os.unlink(file_path)


@pytest.mark.asyncio
async def test_send_prompt_async_no_file():
    # Create a PromptRequestPiece with converted_value that does not point to a valid file.
    request_piece = PromptRequestPiece(role="user", original_value="dummy", converted_value="non_existent_file.pdf")
    prompt_request = PromptRequestResponse(request_pieces=[request_piece])

    # Create HTTPXApiTarget with a dummy transport for non-file mode.
    transport = httpx.MockTransport(dummy_transport_no_file)
    target = HTTPXApiTarget(http_url="http://example.com/data/", method="POST", timeout=180, **{"transport": transport})

    response = await target.send_prompt_async(prompt_request=prompt_request)

    # The dummy transport returns a JSON string containing "Sample JSON response".
    response_text = (
        str(response.request_pieces[0].converted_value) if response.request_pieces[0].converted_value else str(response)
    )
    assert "Sample JSON response" in response_text


@pytest.mark.asyncio
async def test_send_prompt_async_validation():
    # Create an invalid prompt request (empty request_pieces)
    prompt_request = PromptRequestResponse(request_pieces=[])
    target = HTTPXApiTarget(http_url="http://example.com/validate/", method="POST", timeout=180)

    with pytest.raises(ValueError) as excinfo:
        await target.send_prompt_async(prompt_request=prompt_request)

    assert "This target only supports a single prompt request piece." in str(excinfo.value)
