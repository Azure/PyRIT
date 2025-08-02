# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Export and utilities mixin for MemoryInterface containing export and utility operations."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from pyrit.common.path import DB_DATA_PATH
from pyrit.memory.memory_interface.protocol import MemoryInterfaceProtocol

# Use protocol inheritance only during type checking to avoid metaclass conflicts.
# The protocol uses typing._ProtocolMeta which conflicts with the Singleton metaclass
# used by concrete memory classes. This conditional inheritance provides full type
# checking and IDE support while avoiding runtime metaclass conflicts.
if TYPE_CHECKING:
    _MixinBase = MemoryInterfaceProtocol
else:
    _MixinBase = object


class MemoryExportMixin(_MixinBase):
    """Mixin providing export and utility methods for memory management."""

    def export_conversations(
        self,
        *,
        orchestrator_id: Optional[str | uuid.UUID] = None,
        conversation_id: Optional[str | uuid.UUID] = None,
        prompt_ids: Optional[Sequence[str] | Sequence[uuid.UUID]] = None,
        labels: Optional[dict[str, str]] = None,
        sent_after: Optional[datetime] = None,
        sent_before: Optional[datetime] = None,
        original_values: Optional[Sequence[str]] = None,
        converted_values: Optional[Sequence[str]] = None,
        data_type: Optional[str] = None,
        not_data_type: Optional[str] = None,
        converted_value_sha256: Optional[Sequence[str]] = None,
        file_path: Optional[Path] = None,
        export_type: str = "json",
    ) -> Path:
        """
        Exports conversation data with the given inputs to a specified file.
            Defaults to all conversations if no filters are provided.

        Args:
            orchestrator_id (Optional[str | uuid.UUID], optional): The ID of the orchestrator. Defaults to None.
            conversation_id (Optional[str | uuid.UUID], optional): The ID of the conversation. Defaults to None.
            prompt_ids (Optional[Sequence[str] | Sequence[uuid.UUID]], optional): A list of prompt IDs.
                Defaults to None.
            labels (Optional[dict[str, str]], optional): A dictionary of labels. Defaults to None.
            sent_after (Optional[datetime], optional): Filter for prompts sent after this datetime. Defaults to None.
            sent_before (Optional[datetime], optional): Filter for prompts sent before this datetime. Defaults to None.
            original_values (Optional[Sequence[str]], optional): A list of original values. Defaults to None.
            converted_values (Optional[Sequence[str]], optional): A list of converted values. Defaults to None.
            data_type (Optional[str], optional): The data type to filter by. Defaults to None.
            not_data_type (Optional[str], optional): The data type to exclude. Defaults to None.
            converted_value_sha256 (Optional[Sequence[str]], optional): A list of SHA256 hashes of converted values.
                Defaults to None.
            file_path (Optional[Path], optional): The path to the file where the data will be exported.
                Defaults to None.
            export_type (str, optional): The format of the export. Defaults to "json".
        """
        data = self.get_prompt_request_pieces(
            orchestrator_id=orchestrator_id,
            conversation_id=conversation_id,
            prompt_ids=prompt_ids,
            labels=labels,
            sent_after=sent_after,
            sent_before=sent_before,
            original_values=original_values,
            converted_values=converted_values,
            data_type=data_type,
            not_data_type=not_data_type,
            converted_value_sha256=converted_value_sha256,
        )

        # If file_path is not provided, construct a default using the exporter's results_path
        if not file_path:
            file_name = f"exported_conversations_on_{datetime.now().strftime('%Y_%m_%d')}.{export_type}"
            file_path = DB_DATA_PATH / file_name

        self.exporter.export_data(data, file_path=file_path, export_type=export_type)

        return file_path
