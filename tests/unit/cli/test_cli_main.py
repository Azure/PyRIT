# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the PyRIT CLI main module.

Tests command-line argument parsing, scenario listing, and main entry point.
For InitializerRegistry tests, see test_initializer_registry.py.
For ScenarioRegistry tests, see test_scenario_registry.py.
"""

from pathlib import Path
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
    @patch("pyrit.cli.__main__.InitializerRegistry")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    def test_main_list_scenarios_with_scripts(
        self, mock_registry_class, mock_init_registry_class, mock_init_pyrit, capsys
    ):
        """Test main with --list-scenarios and initialization scripts."""
        mock_registry = MagicMock()
        mock_registry.list_scenarios.return_value = []
        mock_registry_class.return_value = mock_registry

        # Mock the resolve_script_paths static method
        mock_init_registry_class.resolve_script_paths.return_value = [Path("/fake/custom.py")]

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
    @patch("pyrit.cli.__main__.InitializerRegistry")
    def test_main_scenario_with_missing_script(
        self, mock_init_registry_class, mock_registry_class, mock_init_pyrit, capsys
    ):
        """Test main with scenario and missing initialization script."""
        # Make resolve_script_paths raise FileNotFoundError
        mock_init_registry_class.resolve_script_paths.side_effect = FileNotFoundError(
            "Initialization script not found: /fake/path/missing.py"
        )

        result = main(["encoding_scenario", "--initialization-scripts", "missing.py"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Initialization script not found" in captured.out

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    @patch("pyrit.cli.__main__.InitializerRegistry")
    def test_main_scenario_with_initializers(
        self, mock_init_registry_class, mock_registry_class, mock_init_pyrit, mock_asyncio_run
    ):
        """Test main with scenario and built-in initializers."""
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        # Mock the resolve_initializer_paths method
        mock_init_registry_instance = MagicMock()
        mock_init_registry_instance.resolve_initializer_paths.return_value = [Path("/fake/simple.py")]
        mock_init_registry_class.return_value = mock_init_registry_instance

        result = main(["encoding_scenario", "--initializers", "simple"])

        assert result == 0
        mock_init_pyrit.assert_called_once()

    @patch("pyrit.cli.__main__.asyncio.run")
    @patch("pyrit.cli.__main__.initialize_pyrit")
    @patch("pyrit.cli.__main__.ScenarioRegistry")
    @patch("pyrit.cli.__main__.InitializerRegistry")
    def test_main_scenario_with_invalid_initializer(
        self, mock_init_registry_class, mock_registry_class, mock_init_pyrit, mock_asyncio_run, capsys
    ):
        """Test main with scenario and invalid initializer name."""
        # Mock the registry instance to raise ValueError
        mock_init_registry_instance = MagicMock()
        mock_init_registry_instance.resolve_initializer_paths.side_effect = ValueError(
            "Built-in initializer 'invalid' not found"
        )
        mock_init_registry_class.return_value = mock_init_registry_instance

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
