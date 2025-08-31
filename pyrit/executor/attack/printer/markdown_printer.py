# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
from typing import List

from IPython.display import Markdown, display

from pyrit.executor.attack.printer.attack_result_printer import AttackResultPrinter
from pyrit.memory import CentralMemory
from pyrit.models import AttackResult, Score
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse


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
        Render the markdown content using appropriate display method.

        Attempts to use IPython.display.Markdown for Jupyter notebook rendering
        when display_inline is True, falling back to print() if not available.

        Args:
            markdown_lines (List[str]): List of markdown strings to render.
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
        """
        Format a score object as markdown with proper styling.

        Converts a Score object into formatted markdown text with appropriate
        emphasis and structure. Handles different score value types and includes
        rationale and metadata when available.

        Args:
            score (Score): The score object to format.
            indent (str): String prefix for indentation. Defaults to "".

        Returns:
            str: Formatted markdown representation of the score.
        """
        lines = []

        # Score value with appropriate formatting
        score_value = score.get_value()
        if isinstance(score_value, bool):
            value_str = str(score_value)
        elif isinstance(score_value, (int, float)):
            value_str = f"**{score_value:.2f}**" if isinstance(score_value, float) else f"**{score_value}**"
        else:
            value_str = f"**{score_value}**"

        lines.append(f"{indent}- **Score Type:** {score.score_type}")
        lines.append(f"{indent}- **Value:** {value_str}")
        lines.append(f"{indent}- **Category:** {score.score_category or 'N/A'}")

        if score.score_rationale:
            # Handle multi-line rationale
            rationale_lines = score.score_rationale.split("\n")
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
        Print the complete attack result as formatted markdown.

        Generates a comprehensive markdown report including attack summary,
        conversation history, scores, and metadata. The output is optimized
        for display in Jupyter notebooks.

        Args:
            result (AttackResult): The attack result to print.
            include_auxiliary_scores (bool): Whether to include auxiliary scores
                in the conversation display. Defaults to False.
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
            result=result, include_auxiliary_scores=include_auxiliary_scores
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
        Print only the conversation history as formatted markdown.

        Extracts and displays the conversation messages from the attack result
        without the summary or metadata sections. Useful for focusing on the
        actual interaction flow.

        Args:
            result (AttackResult): The attack result containing the conversation
                to display.
            include_auxiliary_scores (bool): Whether to include auxiliary scores
                for each message. Defaults to False.
        """
        markdown_lines = await self._get_conversation_markdown_async(
            result=result, include_auxiliary_scores=include_auxiliary_scores
        )
        self._render_markdown(markdown_lines)

    async def print_summary_async(self, result: AttackResult) -> None:
        """
        Print a summary of the attack result as formatted markdown.

        Displays key information about the attack including objective, outcome,
        execution metrics, and final score without the full conversation history.
        Useful for getting a quick overview of the attack results.

        Args:
            result (AttackResult): The attack result to summarize.
        """
        markdown_lines = await self._get_summary_markdown_async(result)
        self._render_markdown(markdown_lines)

    async def _get_conversation_markdown_async(
        self, *, result: AttackResult, include_auxiliary_scores: bool = False
    ) -> List[str]:
        """
        Generate markdown lines for the conversation history.

        Retrieves conversation messages from memory and formats them as markdown,
        organizing by turns and message roles. Handles system messages, user
        inputs, and assistant responses with appropriate formatting.

        Args:
            result (AttackResult): The attack result containing the conversation ID.
            include_auxiliary_scores (bool): Whether to include auxiliary scores
                for each message. Defaults to False.

        Returns:
            List[str]: List of markdown strings representing the formatted
                conversation history.
        """
        markdown_lines = []
        messages = self._memory.get_conversation(conversation_id=result.conversation_id)

        if not messages:
            markdown_lines.append(f"*No conversation found for ID: {result.conversation_id}*\n")
            return markdown_lines

        turn_number = 0

        for message in messages:
            if not message.request_pieces:
                continue

            message_role = message.get_piece().role

            if message_role == "system":
                markdown_lines.extend(self._format_system_message(message))
            elif message_role == "user":
                turn_number += 1
                markdown_lines.extend(await self._format_user_message_async(message=message, turn_number=turn_number))
            else:  # assistant or other response roles
                markdown_lines.extend(await self._format_assistant_message_async(message=message))

            # Add scores if requested
            if include_auxiliary_scores:
                markdown_lines.extend(self._format_message_scores(message))

        return markdown_lines

    def _format_system_message(self, message: PromptRequestResponse) -> List[str]:
        """
        Format a system message as markdown.

        Creates markdown representation of system-level messages, typically
        containing instructions or context for the conversation.

        Args:
            message (PromptRequestResponse): The system message to format.

        Returns:
            List[str]: List of markdown strings representing the system message.
        """
        lines = ["\n### System Message\n"]
        for piece in message.request_pieces:
            lines.append(f"{piece.converted_value}\n")
        return lines

    async def _format_user_message_async(self, *, message: PromptRequestResponse, turn_number: int) -> List[str]:
        """
        Format a user message as markdown with turn numbering.

        Creates markdown representation of user input messages, including turn
        numbers for easy conversation tracking. Shows both original and converted
        values when they differ.

        Args:
            message (PromptRequestResponse): The user message to format.
            turn_number (int): The conversation turn number for this message.

        Returns:
            List[str]: List of markdown strings representing the user message.
        """
        lines = [f"\n### Turn {turn_number}\n", "#### User\n"]

        for piece in message.request_pieces:
            lines.extend(await self._format_piece_content_async(piece=piece, show_original=True))

        return lines

    async def _format_assistant_message_async(self, *, message: PromptRequestResponse) -> List[str]:
        """
        Format an assistant or system response message as markdown.

        Creates markdown representation of response messages from assistants
        or other system components. Automatically capitalizes the role name
        for display purposes.

        Args:
            message (PromptRequestResponse): The response message to format.

        Returns:
            List[str]: List of markdown strings representing the response message.
        """
        lines = []
        role_name = message.request_pieces[0].role.capitalize()

        lines.append(f"\n#### {role_name}\n")

        for piece in message.request_pieces:
            lines.extend(await self._format_piece_content_async(piece=piece, show_original=False))

        return lines

    def _get_audio_mime_type(self, *, audio_path: str) -> str:
        """
        Determine the MIME type for an audio file based on its file extension.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            str: The appropriate MIME type for the audio file.
        """
        if audio_path.lower().endswith(".wav"):
            return "audio/wav"
        elif audio_path.lower().endswith(".ogg"):
            return "audio/ogg"
        elif audio_path.lower().endswith(".m4a"):
            return "audio/mp4"
        else:
            return "audio/mpeg"  # Default fallback for .mp3, .mpeg, and unknown formats

    def _format_image_content(self, *, image_path: str) -> List[str]:
        """
        Format image content as markdown.

        Args:
            image_path (str): The path to the image file.

        Returns:
            List[str]: List of markdown lines for the image.
        """
        return [f"![Image]({image_path})\n"]

    def _format_audio_content(self, *, audio_path: str) -> List[str]:
        """
        Format audio content as HTML5 audio player.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            List[str]: List of markdown lines for the audio player.
        """
        lines = []
        lines.append("<audio controls>")

        audio_type = self._get_audio_mime_type(audio_path=audio_path)

        lines.append(f'<source src="{audio_path}" type="{audio_type}">')
        lines.append("Your browser does not support the audio element.")
        lines.append("</audio>\n")

        return lines

    def _format_error_content(self, *, piece: PromptRequestPiece) -> List[str]:
        """
        Format error response content with proper styling.

        Args:
            piece (PromptRequestPiece): The prompt piece containing the error.

        Returns:
            List[str]: List of markdown lines for the error response.
        """
        lines = []
        lines.append("**Error Response:**\n")
        lines.append(f"*Error Type: {piece.response_error}*\n")
        lines.append("```json")
        lines.append(piece.converted_value)
        lines.append("```\n")

        return lines

    def _format_text_content(self, *, piece: PromptRequestPiece, show_original: bool) -> List[str]:
        """
        Format regular text content.

        Args:
            piece (PromptRequestPiece): The prompt piece containing the text.
            show_original (bool): Whether to show original value if different.

        Returns:
            List[str]: List of markdown lines for the text content.
        """
        lines = []

        if show_original and piece.converted_value != piece.original_value:
            lines.append("**Original:**\n")
            lines.append(f"{piece.original_value}\n")
            lines.append("\n**Converted:**\n")

        lines.append(f"{piece.converted_value}\n")

        return lines

    async def _format_piece_content_async(self, *, piece: PromptRequestPiece, show_original: bool) -> List[str]:
        """
        Format a single piece content based on its data type.

        Handles different content types including text, images, audio, and error responses.

        Args:
            piece (PromptRequestPiece): The prompt piece to format.
            show_original (bool): Whether to show original value if different
                from converted value.

        Returns:
            List[str]: List of markdown lines representing this piece.
        """
        if piece.converted_value_data_type == "image_path":
            return self._format_image_content(image_path=piece.converted_value)
        elif piece.converted_value_data_type == "audio_path":
            return self._format_audio_content(audio_path=piece.converted_value)
        else:
            # Handle text content (including errors)
            if piece.has_error():
                return self._format_error_content(piece=piece)
            else:
                return self._format_text_content(piece=piece, show_original=show_original)

    def _format_message_scores(self, message: PromptRequestResponse) -> List[str]:
        """
        Format scores for all pieces in a message as markdown.

        Retrieves and formats all scores associated with the prompt pieces
        in the given message. Creates a dedicated scores section with
        appropriate markdown formatting.

        Args:
            message (PromptRequestResponse): The message containing pieces
                to format scores for.

        Returns:
            List[str]: List of markdown strings representing the scores.
        """
        lines = []
        for piece in message.request_pieces:
            scores = self._memory.get_prompt_scores(prompt_ids=[str(piece.id)])
            if scores:
                lines.append("\n##### Scores\n")
                for score in scores:
                    lines.append(self._format_score(score, indent=""))
                lines.append("")
        return lines

    async def _get_summary_markdown_async(self, result: AttackResult) -> List[str]:
        """
        Generate markdown lines for the attack summary.

        Creates a comprehensive summary including basic information tables,
        execution metrics, outcome status, and final scores. Uses markdown
        tables for structured data presentation.

        Args:
            result (AttackResult): The attack result to summarize.

        Returns:
            List[str]: List of markdown strings representing the formatted summary.
        """
        markdown_lines = []
        markdown_lines.append("## Attack Summary\n")

        # Basic Information Table
        markdown_lines.append("### Basic Information\n")
        markdown_lines.append("| Field | Value |")
        markdown_lines.append("|-------|-------|")
        markdown_lines.append(f"| **Objective** | {result.objective} |")

        attack_type = result.attack_identifier.get("__type__", "Unknown")

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
