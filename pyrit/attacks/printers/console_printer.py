# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import textwrap
from datetime import datetime
from colorama import Fore, Style, Back

from pyrit.common.display_response import display_image_response
from pyrit.memory import CentralMemory
from pyrit.models import Score, AttackResult, AttackOutcome
from pyrit.attacks.printers import AttackResultPrinter



class ColoredConsoleAttackResultPrinter(AttackResultPrinter):
    """
    Colored console printer for attack results with enhanced formatting.
    
    This printer formats attack results for console display with color coding,
    proper indentation, text wrapping, and visual separators.
    """
    
    def __init__(self, width: int = 100, indent_size: int = 2):
        """
        Initialize the colored console printer.
        
        Args:
            width (int): Maximum width for text wrapping. Must be positive.
                Defaults to 100.
            indent_size (int): Number of spaces for indentation. Must be non-negative.
                Defaults to 2.
        
        Raises:
            ValueError: If width <= 0 or indent_size < 0.
        """
        self._memory = CentralMemory.get_memory_instance()
        self._width = width
        self._indent = " " * indent_size
    
    async def print_result(self, result: AttackResult) -> None:
        """
        Print the complete attack result to console.
        
        This method orchestrates the printing of all components of an attack result,
        including header, summary, conversation history, metadata, and footer.
        
        Args:
            result (AttackResult): The attack result to print. Must not be None.
        """
        # Print header with outcome
        self._print_header(result)
        
        # Print summary information
        await self.print_summary(result)
        
        # Print conversation
        self._print_section_header("Conversation History")
        await self.print_conversation(result)
        
        # Print metadata if available
        if result.metadata:
            self._print_metadata(result.metadata)
        
        # Print footer
        self._print_footer()
    
    async def print_conversation(self, result: AttackResult) -> None:
        """
        Print the conversation history to console with enhanced formatting.
        
        Displays the full conversation between user and assistant, including:
        - Turn numbers
        - Role indicators (USER/ASSISTANT)
        - Original and converted values when different
        - Images if present
        - Scores for each response
        
        Args:
            result (AttackResult): The attack result containing the conversation_id.
                Must have a valid conversation_id attribute.
        """
        messages = self._memory.get_conversation(conversation_id=result.conversation_id)
        
        if not messages:
            print(f"{self._indent}{Fore.YELLOW}⚠ No conversation found for ID: {result.conversation_id}{Style.RESET_ALL}")
            return
        
        turn_number = 0
        for message in messages:
            for piece in message.request_pieces:
                if piece.role == "user":
                    turn_number += 1
                    # User message header
                    print(f"\n{Fore.BLUE}{'─' * self._width}{Style.RESET_ALL}")
                    print(f"{Style.BRIGHT}{Fore.BLUE}🔹 Turn {turn_number} - USER{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}{'─' * self._width}{Style.RESET_ALL}")
                    
                    # Handle converted values
                    if piece.converted_value != piece.original_value:
                        print(f"{Fore.CYAN}{self._indent}📝 Original:{Style.RESET_ALL}")
                        self._print_wrapped_text(piece.original_value, Fore.WHITE)
                        print(f"\n{Fore.CYAN}{self._indent}🔄 Converted:{Style.RESET_ALL}")
                        self._print_wrapped_text(piece.converted_value, Fore.WHITE)
                    else:
                        self._print_wrapped_text(piece.converted_value, Fore.BLUE)
                else:
                    # Assistant message header
                    print(f"\n{Fore.YELLOW}{'─' * self._width}{Style.RESET_ALL}")
                    print(f"{Style.BRIGHT}{Fore.YELLOW}🔸 {piece.role.upper()}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}{'─' * self._width}{Style.RESET_ALL}")
                    
                    self._print_wrapped_text(piece.converted_value, Fore.YELLOW)
                
                # Display images if present
                await display_image_response(piece)
                
                # Print scores with better formatting
                scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(piece.id)])
                if scores:
                    print(f"\n{Style.DIM}{Fore.MAGENTA}{self._indent}📊 Scores:{Style.RESET_ALL}")
                    for score in scores:
                        self._print_score(score)
        
        print(f"\n{Fore.BLUE}{'─' * self._width}{Style.RESET_ALL}")
    
    async def print_summary(self, result: AttackResult) -> None:
        """
        Print a summary of the attack result with enhanced formatting.
        
        Displays:
        - Basic information (objective, attack type, conversation ID)
        - Execution metrics (turns executed, execution time)
        - Outcome information (status, reason)
        - Final score if available
        
        Args:
            result (AttackResult): The attack result to summarize. Must contain
                objective, attack_identifier, conversation_id, executed_turns,
                execution_time_ms, outcome, and optionally outcome_reason and
                last_score attributes.
        """
        self._print_section_header("Attack Summary")
        
        # Basic information
        print(f"{self._indent}{Style.BRIGHT}📋 Basic Information{Style.RESET_ALL}")
        print(f"{self._indent * 2}• Objective: {Fore.CYAN}{result.objective}{Style.RESET_ALL}")
        
        # Extract attack type name from attack_identifier
        attack_type = "Unknown"
        if isinstance(result.attack_identifier, dict) and "__type__" in result.attack_identifier:
            attack_type = result.attack_identifier["__type__"]
        elif isinstance(result.attack_identifier, str):
            attack_type = result.attack_identifier
        
        print(f"{self._indent * 2}• Attack Type: {Fore.CYAN}{attack_type}{Style.RESET_ALL}")
        print(f"{self._indent * 2}• Conversation ID: {Fore.CYAN}{result.conversation_id}{Style.RESET_ALL}")
        
        # Execution metrics
        print(f"\n{self._indent}{Style.BRIGHT}⚡ Execution Metrics{Style.RESET_ALL}")
        print(f"{self._indent * 2}• Turns Executed: {Fore.GREEN}{result.executed_turns}{Style.RESET_ALL}")
        print(f"{self._indent * 2}• Execution Time: {Fore.GREEN}{self._format_time(result.execution_time_ms)}{Style.RESET_ALL}")
        
        # Outcome information
        print(f"\n{self._indent}{Style.BRIGHT}🎯 Outcome{Style.RESET_ALL}")
        outcome_icon = self._get_outcome_icon(result.outcome)
        outcome_color = self._get_outcome_color(result.outcome)
        print(f"{self._indent * 2}• Status: {outcome_color}{outcome_icon} {result.outcome.value.upper()}{Style.RESET_ALL}")
        
        if result.outcome_reason:
            print(f"{self._indent * 2}• Reason: {Fore.WHITE}{result.outcome_reason}{Style.RESET_ALL}")
        
        # Final score
        if result.last_score:
            print(f"\n{self._indent}{Style.BRIGHT}🏆 Final Score{Style.RESET_ALL}")
            self._print_score(result.last_score, indent_level=2)
    
    def _print_header(self, result: AttackResult) -> None:
        """
        Print the header with outcome-based coloring and styling.
        
        Creates a visually prominent header that displays the attack outcome
        with appropriate color coding and icons.
        
        Args:
            result (AttackResult): The attack result containing the outcome.
                Must have an outcome attribute of type AttackOutcome.
        """
        color = self._get_outcome_color(result.outcome)
        icon = self._get_outcome_icon(result.outcome)
        
        print(f"\n{color}{'═' * self._width}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{color}{icon} ATTACK RESULT: {result.outcome.value.upper()} {icon}{Style.RESET_ALL}".center(self._width + 20))
        print(f"{color}{'═' * self._width}{Style.RESET_ALL}")
    
    def _print_footer(self) -> None:
        """
        Print a footer with timestamp.
        
        Displays the current timestamp when the report was generated.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{Style.DIM}{Fore.WHITE}{'─' * self._width}{Style.RESET_ALL}")
        print(f"{Style.DIM}{Fore.WHITE}Report generated at: {timestamp}{Style.RESET_ALL}".center(self._width))
    
    def _print_section_header(self, title: str) -> None:
        """
        Print a section header with consistent styling.
        
        Creates a visually distinct section header with background color
        and separator line.
        
        Args:
            title (str): The title text to display in the section header.
        """
        print(f"\n{Style.BRIGHT}{Back.BLUE}{Fore.WHITE} {title} {Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'─' * self._width}{Style.RESET_ALL}")
    
    def _print_metadata(self, metadata: dict) -> None:
        """
        Print metadata in a formatted way.
        
        Displays key-value pairs from the metadata dictionary in a
        consistent bullet-point format.
        
        Args:
            metadata (dict): Dictionary containing metadata key-value pairs.
                Keys and values should be convertible to strings.
        """
        self._print_section_header("Additional Metadata")
        for key, value in metadata.items():
            print(f"{self._indent}• {key}: {Fore.CYAN}{value}{Style.RESET_ALL}")
    
    def _print_score(self, score: Score, indent_level: int = 3) -> None:
        """
        Print a score with proper formatting.
        
        Displays score information including type, value, and rationale
        with appropriate color coding based on score type.
        
        Args:
            score (Score): Score object to be printed.
            indent_level (int): Number of indent units to apply. Defaults to 3.
        """
        indent = self._indent * indent_level
        print(f"{indent}• Type: {Fore.CYAN}{score.score_type}{Style.RESET_ALL}")
        
        # Determine color based on score type and value
        if score.score_type == "true_false":
            score_color = Fore.GREEN if score.get_value() else Fore.RED
        else:
            score_color = Fore.YELLOW
        
        print(f"{indent}• Value: {score_color}{score.score_value}{Style.RESET_ALL}")
        
        if hasattr(score, 'score_rationale') and score.score_rationale:
            print(f"{indent}• Rationale:")
            # Create a custom wrapper for rationale with proper indentation
            rationale_wrapper = textwrap.TextWrapper(
                width=self._width - len(indent) - 2,  # Adjust width to account for indentation
                initial_indent=indent + "  ",
                subsequent_indent=indent + "  ",
                break_long_words=False,
                break_on_hyphens=False
            )
            # Split by newlines first to preserve them
            lines = score.score_rationale.split('\n')
            for line in lines:
                if line.strip():  # Only wrap non-empty lines
                    wrapped_lines = rationale_wrapper.wrap(line)
                    for wrapped_line in wrapped_lines:
                        print(f"{Fore.WHITE}{wrapped_line}{Style.RESET_ALL}")
                else:  # Print empty lines as-is to preserve formatting
                    print(f"{indent}  {Fore.WHITE}{Style.RESET_ALL}")
    
    def _print_wrapped_text(self, text: str, color: str) -> None:
        """
        Print text with proper wrapping and indentation, preserving newlines.
        
        Wraps long lines while preserving the original line breaks and
        applying consistent indentation and coloring.
        
        Args:
            text (str): The text to print. Can contain newlines.
            color (str): Colorama color constant to apply to the text
                (e.g., Fore.BLUE, Fore.RED).
        """
        # Create a new wrapper for each text to ensure proper width calculation
        text_wrapper = textwrap.TextWrapper(
            width=self._width - len(self._indent),  # Adjust width to account for indentation
            initial_indent="",
            subsequent_indent=self._indent,
            break_long_words=True,  # Allow breaking long words to prevent truncation
            break_on_hyphens=True,
            expand_tabs=False,
            replace_whitespace=False  # Preserve whitespace formatting
        )
        
        # Split by newlines first to preserve them
        lines = text.split('\n')
        for line_num, line in enumerate(lines):
            if line.strip():  # Only wrap non-empty lines
                wrapped_lines = text_wrapper.wrap(line)
                for i, wrapped_line in enumerate(wrapped_lines):
                    if line_num == 0 and i == 0:
                        print(f"{self._indent}{color}{wrapped_line}{Style.RESET_ALL}")
                    else:
                        print(f"{self._indent * 2}{color}{wrapped_line}{Style.RESET_ALL}")
            else:  # Print empty lines as-is to preserve formatting
                print(f"{self._indent}{color}{Style.RESET_ALL}")
    
    def _get_outcome_color(self, outcome: AttackOutcome) -> str:
        """
        Get the color for an outcome.
        
        Maps AttackOutcome enum values to appropriate Colorama color constants.
        
        Args:
            outcome (AttackOutcome): The attack outcome enum value.
        
        Returns:
            str: Colorama color constant (Fore.GREEN, Fore.RED, Fore.YELLOW,
                or Fore.WHITE for unknown outcomes).
        """
        return {
            AttackOutcome.SUCCESS: Fore.GREEN,
            AttackOutcome.FAILURE: Fore.RED,
            AttackOutcome.UNDETERMINED: Fore.YELLOW
        }.get(outcome, Fore.WHITE)
    
    def _get_outcome_icon(self, outcome: AttackOutcome) -> str:
        """
        Get an icon for an outcome.
        
        Maps AttackOutcome enum values to appropriate Unicode emoji icons.
        
        Args:
            outcome (AttackOutcome): The attack outcome enum value.
        
        Returns:
            str: Unicode emoji string ("✅", "❌", "❓", or empty string
                for unknown outcomes).
        """
        return {
            AttackOutcome.SUCCESS: "✅",
            AttackOutcome.FAILURE: "❌",
            AttackOutcome.UNDETERMINED: "❓"
        }.get(outcome, "")
    
    def _format_time(self, milliseconds: int) -> str:
        """
        Format time in a human-readable way.
        
        Converts milliseconds to appropriate units (ms, s, or m+s) based
        on the magnitude of the value.
        
        Args:
            milliseconds (int): Time duration in milliseconds. Should be
                non-negative.
        
        Returns:
            str: Formatted time string (e.g., "500ms", "2.50s", "1m 30s").
        
        Raises:
            TypeError: If milliseconds is not an integer.
            ValueError: If milliseconds is negative.
        """
        if milliseconds < 1000:
            return f"{milliseconds}ms"
        elif milliseconds < 60000:
            return f"{milliseconds / 1000:.2f}s"
        else:
            minutes = milliseconds // 60000
            seconds = (milliseconds % 60000) / 1000
            return f"{minutes}m {seconds:.0f}s"