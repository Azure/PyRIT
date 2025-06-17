# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging

from pyrit.exceptions import (
    AttackExecutionException,
    AttackValidationException,
    BadRequestException,
    EmptyResponseException,
    InvalidJsonException,
    MissingPromptPlaceholderException,
    PyritException,
    RateLimitException,
)


def test_pyrit_exception_initialization():
    ex = PyritException(500, message="Internal Server Error")
    assert ex.status_code == 500
    assert ex.message == "Internal Server Error"
    assert str(ex) == "Status Code: 500, Message: Internal Server Error"


def test_pyrit_exception_process_exception(caplog):
    ex = PyritException(500, message="Internal Server Error")
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()
    assert json.loads(result) == {"status_code": 500, "message": "Internal Server Error"}
    assert "PyritException encountered: Status Code: 500, Message: Internal Server Error" in caplog.text


def test_bad_request_exception_initialization():
    ex = BadRequestException()
    assert ex.status_code == 400
    assert ex.message == "Bad Request"
    assert str(ex) == "Status Code: 400, Message: Bad Request"


def test_rate_limit_exception_initialization():
    ex = RateLimitException()
    assert ex.status_code == 429
    assert ex.message == "Rate Limit Exception"
    assert str(ex) == "Status Code: 429, Message: Rate Limit Exception"


def test_empty_response_exception_initialization():
    ex = EmptyResponseException()
    assert ex.status_code == 204
    assert ex.message == "No Content"
    assert str(ex) == "Status Code: 204, Message: No Content"


def test_invalid_json_exception_initialization():
    ex = InvalidJsonException()
    assert ex.status_code == 500
    assert ex.message == "Invalid JSON Response"
    assert str(ex) == "Status Code: 500, Message: Invalid JSON Response"


def test_bad_request_exception_process_exception(caplog):
    ex = BadRequestException()
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()
    assert json.loads(result) == {"status_code": 400, "message": "Bad Request"}
    assert "BadRequestException encountered: Status Code: 400, Message: Bad Request" in caplog.text


def test_rate_limit_exception_process_exception(caplog):
    ex = RateLimitException()
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()
    assert json.loads(result) == {"status_code": 429, "message": "Rate Limit Exception"}
    assert "RateLimitException encountered: Status Code: 429, Message: Rate Limit Exception" in caplog.text


def test_empty_response_exception_process_exception(caplog):
    ex = EmptyResponseException()
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()
    assert json.loads(result) == {"status_code": 204, "message": "No Content"}
    assert "EmptyResponseException encountered: Status Code: 204, Message: No Content" in caplog.text


def test_empty_prompt_placeholder_exception(caplog):
    ex = MissingPromptPlaceholderException()
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()
    assert json.loads(result) == {"status_code": 500, "message": "No prompt placeholder"}
    assert (
        "MissingPromptPlaceholderException encountered: Status Code: 500, Message: No prompt placeholder" in caplog.text
    )


def test_remove_markdown_json_exception(caplog):
    ex = InvalidJsonException()
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()
    assert json.loads(result) == {"status_code": 500, "message": "Invalid JSON Response"}
    assert "InvalidJsonException encountered: Status Code: 500, Message: Invalid JSON Response" in caplog.text


def test_attack_validation_exception_initialization():
    ex = AttackValidationException()
    assert ex.status_code == 400
    assert ex.message == "Attack context validation failed"
    assert ex.context_info == {}
    assert str(ex) == "Status Code: 400, Message: Attack context validation failed"


def test_attack_validation_exception_with_context():
    context_info = {"attack_type": "SampleAttack", "error_type": "ValueError"}
    ex = AttackValidationException(message="Custom validation error", context_info=context_info)
    assert ex.status_code == 400
    assert ex.message == "Custom validation error"
    assert ex.context_info == context_info


def test_attack_validation_exception_process_exception(caplog):
    context_info = {"attack_type": "SampleAttack", "original_error": "Missing objective"}
    ex = AttackValidationException(message="Validation failed", context_info=context_info)
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()

    expected_result = {
        "status_code": 400,
        "message": "Validation failed",
        "context_info": context_info,
    }
    assert json.loads(result) == expected_result
    assert "AttackValidationException encountered:" in caplog.text
    assert "Status Code: 400" in caplog.text
    assert "Message: Validation failed" in caplog.text
    assert "Context: {'attack_type': 'SampleAttack', 'original_error': 'Missing objective'}" in caplog.text


def test_attack_execution_exception_initialization():
    ex = AttackExecutionException()
    assert ex.status_code == 500
    assert ex.message == "Attack execution failed"
    assert ex.attack_name is None
    assert ex.objective is None
    assert str(ex) == "Status Code: 500, Message: Attack execution failed"


def test_attack_execution_exception_with_details():
    ex = AttackExecutionException(
        message="Custom execution error",
        attack_name="SampleAttack",
        objective="sample objective",
    )
    assert ex.status_code == 500
    assert ex.message == "Custom execution error"
    assert ex.attack_name == "SampleAttack"
    assert ex.objective == "sample objective"


def test_attack_execution_exception_process_exception(caplog):
    ex = AttackExecutionException(
        message="Attack failed unexpectedly",
        attack_name="SampleAttack",
        objective="sample objective",
    )
    with caplog.at_level(logging.ERROR):
        result = ex.process_exception()

    expected_result = {
        "status_code": 500,
        "message": "Attack failed unexpectedly",
        "attack_name": "SampleAttack",
        "objective": "sample objective",
    }
    assert json.loads(result) == expected_result
    assert "AttackExecutionException encountered:" in caplog.text
    assert "Status Code: 500" in caplog.text
    assert "Message: Attack failed unexpectedly" in caplog.text
    assert "Attack: SampleAttack" in caplog.text
    assert "Objective: sample objective" in caplog.text
