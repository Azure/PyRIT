# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
from typing import List
from IPython.display import Markdown, display

from pyrit.common.display_response import display_image_response
from pyrit.executor.attack.printer.attack_result_printer import AttackResultPrinter
from pyrit.memory import CentralMemory
from pyrit.models import AttackResult, Score


class MarkdownAttackResultPrinter(AttackResultPrinter):
    """
    Markdown printer for attack results optimized for Jupyter notebooks.

    This printer formats attack results as markdown, making them ideal for display
    in Jupyter notebooks where LLM responses often contain code blocks and other
    markdown formatting that should be properly rendered.
    """

    def __init__(self, *, display_inline: bool = True):
        """
        Initialize the markdown printer.

        Args:
            display_inline (bool): If True, uses IPython.display to render markdown
                inline in Jupyter notebooks. If False, prints markdown strings.
                Defaults to True.
        """
        self._memory = CentralMemory.get_memory_instance()
        self._display_inline = display_inline

    def _render_markdown(self, markdown_lines: List[str]) -> None:
        """
        Render the markdown content.
        
        Args:
            markdown_lines: List of markdown strings to render.
        """
        full_markdown = "\n".join(markdown_lines)
        
        if self._display_inline:
            try:
                display(Markdown(full_markdown))
            except (ImportError, NameError):
                # Fallback to print if not in Jupyter environment
                print(full_markdown)
        else:
            print(full_markdown)

    def _format_score(self, score: Score, indent: str = "") -> str:
        """Format a score object as markdown."""
        lines = []
        
        # Score value with appropriate formatting
        score_value = score.get_value()
        if isinstance(score_value, bool):
            value_str = "True" if score_value else "False"
        elif isinstance(score_value, (int, float)):
            value_str = f"**{score_value:.2f}**" if isinstance(score_value, float) else f"**{score_value}**"
        else:
            value_str = f"**{score_value}**"
        
        lines.append(f"{indent}- **Score Type:** {score.score_type}")
        lines.append(f"{indent}- **Value:** {value_str}")
        lines.append(f"{indent}- **Category:** {score.score_category or 'N/A'}")
        
        if score.score_rationale:
            # Handle multi-line rationale
            rationale_lines = score.score_rationale.split('\n')
            if len(rationale_lines) > 1:
                lines.append(f"{indent}- **Rationale:**")
                for line in rationale_lines:
                    lines.append(f"{indent}  {line}")
            else:
                lines.append(f"{indent}- **Rationale:** {score.score_rationale}")
        
        if score.score_metadata:
            lines.append(f"{indent}- **Metadata:** `{score.score_metadata}`")
        
        return "\n".join(lines)

    async def print_result_async(self, result: AttackResult, *, include_auxiliary_scores: bool = False) -> None:
        """
        Print the complete attack result as markdown.

        Args:
            result (AttackResult): The attack result to print.
            include_auxiliary_scores (bool): Whether to include auxiliary scores.
                Defaults to False.
        """
        markdown_lines = []
        
        # Header with outcome
        outcome_emoji = self._get_outcome_icon(result.outcome)
        markdown_lines.append(f"# {outcome_emoji} Attack Result: {result.outcome.value.upper()}\n")
        markdown_lines.append("---\n")

        # Summary section
        summary_lines = await self._get_summary_markdown_async(result)
        markdown_lines.extend(summary_lines)
        markdown_lines.append("---\n")

        # Conversation history
        markdown_lines.append("\n## Conversation History\n")
        conversation_lines = await self._get_conversation_markdown_async(
            result, include_auxiliary_scores=include_auxiliary_scores
        )
        markdown_lines.extend(conversation_lines)

        # Metadata if available
        if result.metadata:
            markdown_lines.append("\n## Additional Metadata\n")
            for key, value in result.metadata.items():
                # Only include metadata that can be converted to string
                try:
                    # Try to convert to string
                    str_value = str(value)
                    markdown_lines.append(f"- **{key}:** {str_value}")
                except Exception:
                    # Skip values that can't be stringified
                    pass

        # Footer
        markdown_lines.append("\n---")
        markdown_lines.append(f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        self._render_markdown(markdown_lines)

    async def print_conversation_async(self, result: AttackResult, *, include_auxiliary_scores: bool = False) -> None:
        """
        Print only the conversation history as markdown.

        Args:
            result (AttackResult): The attack result containing the conversation.
            include_auxiliary_scores (bool): Whether to include auxiliary scores.
                Defaults to False.
        """
        markdown_lines = await self._get_conversation_markdown_async(
            result, include_auxiliary_scores=include_auxiliary_scores
        )
        self._render_markdown(markdown_lines)

    async def print_summary_async(self, result: AttackResult) -> None:
        """
        Print a summary of the attack result as markdown.

        Args:
            result (AttackResult): The attack result to summarize.
        """
        markdown_lines = await self._get_summary_markdown_async(result)
        self._render_markdown(markdown_lines)

    async def _get_conversation_markdown_async(
        self, result: AttackResult, *, include_auxiliary_scores: bool = False
    ) -> List[str]:
        """
        Generate markdown lines for the conversation history.

        Args:
            result (AttackResult): The attack result containing the conversation.
            include_auxiliary_scores (bool): Whether to include auxiliary scores.

        Returns:
            List[str]: List of markdown strings representing the conversation.
        """
        markdown_lines = []
        messages = self._memory.get_conversation(conversation_id=result.conversation_id)

        if not messages:
            markdown_lines.append(f"*No conversation found for ID: {result.conversation_id}*\n")
            return markdown_lines

        turn_number = 0
        current_turn_has_user = False
        
        for message in messages:
            for piece in message.request_pieces:
                if piece.role == "system":
                    # System message
                    markdown_lines.append("\n### System Message\n")
                    markdown_lines.append(f"{piece.converted_value}\n")
                    
                elif piece.role == "user":
                    turn_number += 1
                    current_turn_has_user = True
                    # Start new turn
                    markdown_lines.append(f"\n### Turn {turn_number}\n")
                    markdown_lines.append("#### User\n")
                    
                    # Show original and converted if different
                    if piece.converted_value != piece.original_value:
                        markdown_lines.append("**Original:**\n")
                        markdown_lines.append(f"{piece.original_value}\n")
                        markdown_lines.append("\n**Converted:**\n")
                        markdown_lines.append(f"{piece.converted_value}\n")
                    else:
                        # Display content as markdown
                        markdown_lines.append(f"{piece.converted_value}\n")
                    
                else:
                    # Assistant/Model response
                    # Only add Turn header if we haven't seen a user message in this turn
                    if not current_turn_has_user:
                        turn_number += 1
                        markdown_lines.append(f"\n### Turn {turn_number}\n")
                    
                    markdown_lines.append(f"\n#### {piece.role.capitalize()}\n")
                    current_turn_has_user = False
                    
                    # Display response as markdown
                    response_text = piece.converted_value
                    markdown_lines.append(f"{response_text}\n")

                # Display images if present
                try:
                    await display_image_response(piece)
                except Exception:
                    # Image display might not work outside Jupyter
                    pass

                # Display scores if requested
                if include_auxiliary_scores:
                    scores = self._memory.get_prompt_scores(prompt_ids=[str(piece.id)])
                    if scores:
                        markdown_lines.append("\n##### Scores\n")
                        for score in scores:
                            markdown_lines.append(self._format_score(score, indent=""))
                        markdown_lines.append("")

        return markdown_lines

    async def _get_summary_markdown_async(self, result: AttackResult) -> List[str]:
        """
        Generate markdown lines for the attack summary.

        Args:
            result (AttackResult): The attack result to summarize.

        Returns:
            List[str]: List of markdown strings representing the summary.
        """
        markdown_lines = []
        markdown_lines.append("## Attack Summary\n")
        
        # Basic Information Table
        markdown_lines.append("### Basic Information\n")
        markdown_lines.append("| Field | Value |")
        markdown_lines.append("|-------|-------|")
        markdown_lines.append(f"| **Objective** | {result.objective} |")
        
        # Extract attack type
        attack_type = "Unknown"
        if isinstance(result.attack_identifier, dict) and "__type__" in result.attack_identifier:
            attack_type = result.attack_identifier["__type__"]
        elif isinstance(result.attack_identifier, str):
            attack_type = result.attack_identifier
        
        markdown_lines.append(f"| **Attack Type** | `{attack_type}` |")
        markdown_lines.append(f"| **Conversation ID** | `{result.conversation_id}` |")
        
        # Execution Metrics
        markdown_lines.append("\n### Execution Metrics\n")
        markdown_lines.append("| Metric | Value |")
        markdown_lines.append("|--------|-------|")
        markdown_lines.append(f"| **Turns Executed** | {result.executed_turns} |")
        markdown_lines.append(f"| **Execution Time** | {self._format_time(result.execution_time_ms)} |")
        
        # Outcome
        outcome_emoji = self._get_outcome_icon(result.outcome)
        markdown_lines.append("\n### Outcome\n")
        markdown_lines.append(f"**Status:** {outcome_emoji} **{result.outcome.value.upper()}**\n")
        
        if result.outcome_reason:
            markdown_lines.append(f"**Reason:** {result.outcome_reason}\n")
        
        # Final Score
        if result.last_score:
            markdown_lines.append("\n### Final Score\n")
            markdown_lines.append(self._format_score(result.last_score))

        return markdown_lines