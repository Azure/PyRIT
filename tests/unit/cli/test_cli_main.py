# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the PyRIT CLI main module.
"""

from unittest.mock import MagicMock, patch

import pytest

from pyrit.cli.__main__ import list_scenarios, main, parse_args


class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_parse_args_list_scenarios(self):
        """Test parsing --list-scenarios flag."""
        args = parse_args(["--list-scenarios"])
        assert args.list_scenarios is True
        assert args.verbose is False

    def test_parse_args_list_initializers(self):
        """Test parsing --list-initializers flag."""
        args = parse_args(["--list-initializers"])
        assert args.list_initializers is True
        assert args.verbose is False

    def test_parse_args_scenario_basic(self):
        """Test parsing with just scenario name."""
        args = parse_args(["encoding_scenario"])
        assert args.scenario_name == "encoding_scenario"
        assert args.database == "SQLite"
        assert args.verbose is False

    def test_parse_args_scenario_with_database(self):
        """Test parsing scenario with database option."""
        args = parse_args(["foundry_scenario", "--database", "InMemory"])
        assert args.scenario_name == "foundry_scenario"
        assert args.database == "InMemory"

    def test_parse_args_scenario_with_initializers(self):
        """Test parsing scenario with initializers option."""
        args = parse_args(["encoding_scenario", "--initializers", "simple", "airt"])
        assert args.scenario_name == "encoding_scenario"
        assert args.initializers == ["simple", "airt"]

    def test_parse_args_scenario_with_dot_notation_initializers(self):
        """Test parsing scenario with dot notation initializers."""
        args = parse_args(["foundry_scenario", "--initializers", "scenarios.objective_target"])
        assert args.scenario_name == "foundry_scenario"
        assert args.initializers == ["scenarios.objective_target"]

    def test_parse_args_scenario_with_initialization_scripts(self):
        """Test parsing scenario with initialization scripts."""
        args = parse_args(["foundry_scenario", "--initialization-scripts", "script1.py", "script2.py"])
        assert args.scenario_name == "foundry_scenario"
        assert args.initialization_scripts == ["script1.py", "script2.py"]

    def test_parse_args_scenario_all_options(self):
        """Test parsing scenario with all options."""
        args = parse_args(
            [
                "--verbose",
                "encoding_scenario",
                "--database",
                "AzureSQL",
                "--initializers",
                "simple",
                "--initialization-scripts",
                "custom.py",
            ]
        )
        assert args.scenario_name == "encoding_scenario"
        assert args.database == "AzureSQL"
        assert args.initializers == ["simple"]
        assert args.initialization_scripts == ["custom.py"]
        assert args.verbose is True

    def test_parse_args_no_scenario(self):
        """Test parsing with no scenario name."""
        args = parse_args([])
        assert args.scenario_name is None

    def test_parse_args_invalid_database(self):
        """Test parsing with invalid database choice."""
        with pytest.raises(SystemExit):
            parse_args(["encoding_scenario", "--database", "InvalidDB"])

    def test_parse_args_list_scenarios_with_initialization_scripts(self):
        """Test parsing --list-scenarios with initialization scripts."""
        args = parse_args(["--list-scenarios", "--initialization-scripts", "custom.py"])
        assert args.list_scenarios is True
        assert args.initialization_scripts == ["custom.py"]


class TestListScenarios:
    """Tests for list_scenarios function."""

    def test_list_scenarios_no_scenarios(self, capsys):
        """Test list_scenarios with no scenarios available."""
        registry = MagicMock()
        registry.list_scenarios.return_value = []

        list_scenarios(registry=registry)

        captured = capsys.readouterr()
        assert "No scenarios found" in captured.out

    def test_list_scenarios_with_scenarios(self, capsys):
        """Test list_scenarios with available scenarios."""
        registry = MagicMock()
        registry.list_scenarios.return_value = [
            {
                "name": "encoding_scenario",
                "class_name": "EncodingScenario",
                "description": "Test encoding attacks on AI models",
            },
            {
                "name": "foundry_scenario",
                "class_name": "FoundryScenario",
                "description": "Comprehensive security testing scenario",
            },
        ]

        list_scenarios(registry=registry)

        captured = capsys.readouterr()
        assert "Available Scenarios:" in captured.out
        assert "encoding_scenario" in captured.out
        assert "EncodingScenario" in captured.out
        assert "foundry_scenario" in captured.out
        assert "FoundryScenario" in captured.out
        assert "Total scenarios: 2" in captured.out


class TestMain:
    """Tests for main function."""

    def test_main_no_scenario(self, capsys):
        """Test main with no scenario returns error."""
        result = main([])
        assert result == 1

        captured = capsys.readouterr()
        assert "Error: No scenario specified" in captured.out

    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_list_scenarios(self, mock_registry_class, capsys):
        """Test main with --list-scenarios flag."""
        mock_registry = MagicMock()
        mock_registry.list_scenarios.return_value = [
            {
                "name": "encoding_scenario",
                "class_name": "EncodingScenario",
                "description": "Test scenario",
            }
        ]
        mock_registry_class.return_value = mock_registry

        result = main(["--list-scenarios"])

        assert result == 0
        mock_registry_class.assert_called_once()
        mock_registry.list_scenarios.assert_called_once()

    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.Path")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_list_scenarios_with_scripts(self, mock_registry_class, mock_path_class, mock_init_pyrit, capsys):
        """Test main with --list-scenarios and initialization scripts."""
        mock_registry = MagicMock()
        mock_registry.list_scenarios.return_value = []
        mock_registry_class.return_value = mock_registry

        # Mock path to exist
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_absolute.return_value = False
        mock_path_class.return_value = mock_path_instance
        mock_path_class.cwd.return_value = mock_path_instance

        result = main(["--list-scenarios", "--initialization-scripts", "custom.py"])

        assert result == 0
        mock_init_pyrit.assert_called_once()
        mock_registry.discover_user_scenarios.assert_called_once()

    @patch("pyrit.cli.__main__.list_initializers")
    def test_main_list_initializers(self, mock_list_init):
        """Test main with --list-initializers flag."""
        result = main(["--list-initializers"])

        assert result == 0
        mock_list_init.assert_called_once()

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_scenario_command(self, mock_registry_class, mock_init_pyrit, mock_asyncio_run):
        """Test main with scenario name."""
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        result = main(["encoding_scenario"])

        assert result == 0
        mock_init_pyrit.assert_called_once()
        mock_registry_class.assert_called_once()
        mock_asyncio_run.assert_called_once()

    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_scenario_with_missing_script(self, mock_registry_class, mock_init_pyrit, capsys):
        """Test main with scenario and missing initialization script."""
        with patch("pyrit.cli.__main__.Path") as mock_path_class:
            # Create a mock path instance that says the file doesn't exist
            mock_script_path = MagicMock()
            mock_script_path.exists.return_value = False
            mock_script_path.is_absolute.return_value = False
            mock_script_path.absolute.return_value = "C:\\fake\\path\\missing.py"

            # Mock Path() to return our script path
            mock_path_class.return_value = mock_script_path

            # Mock Path.cwd() to return a base path
            mock_cwd = MagicMock()
            mock_path_class.cwd.return_value = mock_cwd
            mock_cwd.__truediv__ = lambda self, other: mock_script_path

            result = main(["encoding_scenario", "--initialization-scripts", "missing.py"])

            assert result == 1
            captured = capsys.readouterr()
            assert "Initialization script not found" in captured.out

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.Path")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_scenario_with_initializers(
        self, mock_registry_class, mock_init_pyrit, mock_path_class, mock_asyncio_run
    ):
        """Test main with scenario and built-in initializers."""
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        # Mock the Path object and its methods
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

        with patch("pyrit.common.path.PYRIT_PATH", "/fake/pyrit"):
            result = main(["encoding_scenario", "--initializers", "simple"])

        assert result == 0
        mock_init_pyrit.assert_called_once()

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_scenario_with_invalid_initializer(
        self, mock_registry_class, mock_init_pyrit, mock_asyncio_run, capsys
    ):
        """Test main with scenario and invalid initializer name."""

        # Mock Path to make the invalid initializer file not exist
        with patch("pyrit.common.path.PYRIT_PATH", "/fake/pyrit"):
            with patch("pyrit.cli.__main__.Path") as mock_path_class:
                # Create a mock that handles the path chain properly
                mock_initializer_file = MagicMock()
                mock_initializer_file.exists.return_value = False

                # Make the Path() call and all divisions return objects that support further division
                # Path(PYRIT_PATH) -> ... / "setup" -> ... / "initializers" -> ... / "invalid.py" -> mock_file
                def path_truediv(self_arg, other):
                    if other == "invalid.py":
                        return mock_initializer_file
                    # Return a mock that also supports truediv for chaining
                    next_mock = MagicMock()
                    next_mock.__truediv__ = path_truediv
                    next_mock.with_suffix = lambda x: mock_initializer_file
                    return next_mock

                mock_path_instance = MagicMock()
                mock_path_instance.__truediv__ = path_truediv
                mock_path_class.return_value = mock_path_instance

                result = main(["encoding_scenario", "--initializers", "invalid"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Built-in initializer 'invalid' not found" in captured.out

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_scenario_with_database_option(self, mock_registry_class, mock_init_pyrit, mock_asyncio_run):
        """Test main with scenario and database option."""
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        result = main(["encoding_scenario", "--database", "InMemory"])

        assert result == 0
        # Verify initialize_pyrit was called with InMemory
        call_args = mock_init_pyrit.call_args
        assert call_args[1]["memory_db_type"] == "InMemory"

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_scenario_exception_handling(self, mock_registry_class, mock_init_pyrit, mock_asyncio_run, capsys):
        """Test main handles exceptions properly."""
        mock_init_pyrit.side_effect = Exception("Test error")

        result = main(["encoding_scenario"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.out
