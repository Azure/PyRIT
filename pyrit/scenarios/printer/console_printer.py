# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from colorama import Fore, Style

from pyrit.scenarios.printer.scenario_result_printer import ScenarioResultPrinter
from pyrit.scenarios.scenario_result import ScenarioResult


class ConsoleScenarioResultPrinter(ScenarioResultPrinter):
    """
    Console printer for scenario results with enhanced formatting.

    This printer formats scenario results for console display with optional color coding,
    proper indentation, and visual separators. Colors can be disabled for consoles
    that don't support ANSI characters.
    """

    def __init__(self, *, width: int = 100, indent_size: int = 2, enable_colors: bool = True):
        """
        Initialize the console printer.

        Args:
            width (int): Maximum width for text wrapping. Must be positive.
                Defaults to 100.
            indent_size (int): Number of spaces for indentation. Must be non-negative.
                Defaults to 2.
            enable_colors (bool): Whether to enable ANSI color output. When False,
                all output will be plain text without colors. Defaults to True.

        Raises:
            ValueError: If width <= 0 or indent_size < 0.
        """
        self._width = width
        self._indent = " " * indent_size
        self._enable_colors = enable_colors

    def _print_colored(self, text: str, *colors: str) -> None:
        """
        Print text with color formatting if colors are enabled.

        Args:
            text (str): The text to print.
            *colors: Variable number of colorama color constants to apply.
        """
        if self._enable_colors and colors:
            color_prefix = "".join(colors)
            print(f"{color_prefix}{text}{Style.RESET_ALL}")
        else:
            print(text)

    def _print_section_header(self, title: str) -> None:
        """
        Print a section header with visual separation.

        Args:
            title (str): The section title to display.
        """
        print()
        self._print_colored(f"▼ {title}", Style.BRIGHT, Fore.CYAN)
        self._print_colored("─" * self._width, Fore.CYAN)

    async def print_summary_async(self, result: ScenarioResult) -> None:
        """
        Print a summary of the scenario result with per-strategy breakdown.

        Displays:
        - Scenario identification (name, version, PyRIT version)
        - Target and scorer information
        - Overall statistics
        - Per-strategy success rates and result counts

        Args:
            result (ScenarioResult): The scenario result to summarize
        """
        # Print header
        self._print_header(result)

        # Scenario information
        self._print_section_header("Scenario Information")
        self._print_colored(f"{self._indent}📋 Scenario Details", Style.BRIGHT)
        self._print_colored(f"{self._indent * 2}• Name: {result.scenario_identifier.name}", Fore.CYAN)
        self._print_colored(f"{self._indent * 2}• Scenario Version: {result.scenario_identifier.version}", Fore.CYAN)
        self._print_colored(f"{self._indent * 2}• Description: {result.scenario_identifier.description}", Fore.CYAN)
        self._print_colored(f"{self._indent * 2}• PyRIT Version: {result.scenario_identifier.pyrit_version}", Fore.CYAN)

        # Target information
        print()
        self._print_colored(f"{self._indent}🎯 Target Information", Style.BRIGHT)
        target_type = result.objective_target_identifier.get("__type__", "Unknown")
        target_model = result.objective_target_identifier.get("model_name", "Unknown")
        target_endpoint = result.objective_target_identifier.get("endpoint", "Unknown")

        self._print_colored(f"{self._indent * 2}• Target Type: {target_type}", Fore.CYAN)
        self._print_colored(f"{self._indent * 2}• Target Model: {target_model}", Fore.CYAN)
        self._print_colored(f"{self._indent * 2}• Target Endpoint: {target_endpoint}", Fore.CYAN)

        # Scorer information
        if result.objective_scorer_identifier:
            print()
            self._print_colored(f"{self._indent}📊 Scorer Information", Style.BRIGHT)
            self._print_scorer_info(result.objective_scorer_identifier, indent_level=2)

        # Overall statistics
        self._print_section_header("Overall Statistics")
        total_results = sum(len(results) for results in result.attack_results.values())
        total_strategies = len(result.get_strategies_used())
        overall_rate = result.objective_achieved_rate()

        self._print_colored(f"{self._indent}📈 Summary", Style.BRIGHT)
        self._print_colored(f"{self._indent * 2}• Total Strategies: {total_strategies}", Fore.GREEN)
        self._print_colored(f"{self._indent * 2}• Total Attack Results: {total_results}", Fore.GREEN)
        self._print_colored(
            f"{self._indent * 2}• Overall Success Rate: {overall_rate}%", self._get_rate_color(overall_rate)
        )

        objectives = result.get_objectives()
        self._print_colored(f"{self._indent * 2}• Unique Objectives: {len(objectives)}", Fore.GREEN)

        # Per-strategy breakdown
        self._print_section_header("Per-Strategy Breakdown")
        strategies = result.get_strategies_used()

        for strategy in strategies:
            results_for_strategy = result.attack_results[strategy]
            strategy_rate = result.objective_achieved_rate(attack_run_name=strategy)

            print()
            self._print_colored(f"{self._indent}🔸 Strategy: {strategy}", Style.BRIGHT)
            self._print_colored(f"{self._indent * 2}• Number of Results: {len(results_for_strategy)}", Fore.YELLOW)
            self._print_colored(
                f"{self._indent * 2}• Success Rate: {strategy_rate}%", self._get_rate_color(strategy_rate)
            )

        # Print footer
        self._print_footer()

    def _print_header(self, result: ScenarioResult) -> None:
        """
        Print the header with scenario name.

        Args:
            result (ScenarioResult): The scenario result.
        """
        print()
        self._print_colored("═" * self._width, Fore.CYAN)
        header_text = f"📊 SCENARIO RESULTS: {result.scenario_identifier.name}"
        self._print_colored(header_text.center(self._width), Style.BRIGHT, Fore.CYAN)
        self._print_colored("═" * self._width, Fore.CYAN)

    def _print_footer(self) -> None:
        """
        Print a footer separator.
        """
        print()
        self._print_colored("═" * self._width, Fore.CYAN)
        print()

    def _print_scorer_info(self, scorer_identifier: dict, *, indent_level: int = 2) -> None:
        """
        Print scorer information including nested sub-scorers.

        Args:
            scorer_identifier (dict): The scorer identifier dictionary.
            indent_level (int): Current indentation level for nested display.
        """
        scorer_type = scorer_identifier.get("__type__", "Unknown")
        indent = self._indent * indent_level

        self._print_colored(f"{indent}• Scorer Type: {scorer_type}", Fore.CYAN)

        # Check for sub_identifier
        sub_identifier = scorer_identifier.get("sub_identifier")
        if sub_identifier:
            # Handle list of sub-scorers (composite scorer)
            if isinstance(sub_identifier, list):
                self._print_colored(f"{indent}  └─ Composite of {len(sub_identifier)} scorer(s):", Fore.CYAN)
                for sub_scorer in sub_identifier:
                    self._print_scorer_info(sub_scorer, indent_level=indent_level + 3)
            # Handle single nested scorer
            elif isinstance(sub_identifier, dict):
                self._print_colored(f"{indent}  └─ Wraps:", Fore.CYAN)
                self._print_scorer_info(sub_identifier, indent_level=indent_level + 2)

    def _get_rate_color(self, rate: int) -> str:
        """
        Get color based on success rate.

        Args:
            rate (int): Success rate percentage (0-100)

        Returns:
            str: Colorama color constant
        """
        if rate >= 75:
            return Fore.RED  # High success (bad for security)
        elif rate >= 50:
            return Fore.YELLOW  # Medium success
        elif rate >= 25:
            return Fore.CYAN  # Low success
        else:
            return Fore.GREEN  # Very low success (good for security)
