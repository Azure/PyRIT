# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from mock_alchemy.mocking import UnifiedAlchemyMagicMock
import os
from pathlib import Path
import pytest
from unittest import mock

from pyrit.common.initialize_pyrit import _load_environment_files, initialize_pyrit


@mock.patch("dotenv.load_dotenv")
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.getLogger")
def test_load_environment_files_base_only(mock_logger, mock_exists, mock_load_dotenv):
    # Get the current home path dynamically
    home_path = Path.home()
    _ = home_path / ".env"

    # Mock .env exists, .env.local does not
    mock_exists.side_effect = [True, False]
    mock_logger.return_value = mock.Mock()

    _load_environment_files()

    assert mock_load_dotenv.call_count == 1


@mock.patch("dotenv.load_dotenv")
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.getLogger")
def test_load_environment_files_base_and_local(mock_logger, mock_exists, mock_load_dotenv):
    home_path = Path.home()
    _ = str(home_path / ".env")
    _ = str(home_path / ".env.local")

    mock_exists.side_effect = [True, True]
    mock_logger.return_value = mock.Mock()

    _load_environment_files()

    assert mock_load_dotenv.call_count == 2


@mock.patch("dotenv.load_dotenv")
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.getLogger")
def test_load_environment_files_no_base_no_local(mock_logger, mock_exists, mock_load_dotenv):
    mock_exists.side_effect = [False, False]
    mock_logger.return_value = mock.Mock()

    _load_environment_files()

    mock_load_dotenv.assert_called_once()
    mock_logger.return_value.info.assert_not_called()


@mock.patch("dotenv.load_dotenv")
@mock.patch("pathlib.Path.exists")
def test_load_environment_files_override(mock_exists, mock_load_dotenv):
    home_path = Path.home()
    base_file_path = str(home_path / ".env")
    _ = str(home_path / ".env.local")

    # Mock both .env and .env.local exist
    mock_exists.side_effect = [True, True]

    # Simulate environment variables in base .env and .env.local
    mock_load_dotenv.side_effect = lambda path, override: os.environ.update(
        {
            "TEST_VAR": "base_value" if path == base_file_path else "local_value",
            "COMMON_VAR": "base_common" if path == base_file_path else "local_common",
        }
    )

    # Run the function
    _load_environment_files()

    # Check that variables from .env.local override those in .env
    assert os.getenv("TEST_VAR") == "local_value"
    assert os.getenv("COMMON_VAR") == "local_common"


@pytest.mark.parametrize(
    "memory_db_type,memory_instance_kwargs",
    [
        ("InMemory", {"verbose": True}),
        ("DuckDB", {"verbose": True}),
        (
            "AzureSQL",
            {
                "connection_string": "mssql+pyodbc://test:test@test/test?driver=ODBC+Driver+18+for+SQL+Server",
                "results_container_url": "https://test.blob.core.windows.net/test",
                "results_sas_token": "valid_sas_token",
            },
        ),
    ],
)
@mock.patch("pyrit.memory.central_memory.CentralMemory.set_memory_instance")
@mock.patch("pyrit.common.initialize_pyrit._load_environment_files")
def test_initialize_pyrit(mock_load_env_files, mock_set_memory, memory_db_type, memory_instance_kwargs):
    with (
        mock.patch("pyrit.memory.AzureSQLMemory.get_session") as get_session_mock,
        mock.patch("pyrit.memory.AzureSQLMemory._create_auth_token") as create_auth_token_mock,
        mock.patch("pyrit.memory.AzureSQLMemory._enable_azure_authorization") as enable_azure_authorization_mock,
    ):
        # Mocked for AzureSQL
        session_mock = UnifiedAlchemyMagicMock()
        session_mock.__enter__.return_value = session_mock
        session_mock.is_modified.return_value = True
        get_session_mock.return_value = session_mock

        create_auth_token_mock.return_value = "token"
        enable_azure_authorization_mock.return_value = None

        initialize_pyrit(memory_db_type=memory_db_type, **memory_instance_kwargs)

    mock_load_env_files.assert_called_once()
    mock_set_memory.assert_called_once()


def test_initialize_pyrit_type_check_throws():
    with pytest.raises(ValueError):
        initialize_pyrit(memory_db_type="InvalidType")