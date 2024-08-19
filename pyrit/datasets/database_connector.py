# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional
from uuid import UUID

from pyrit.models import PromptDataset, PromptTemplate


class DatabaseConnector:
    def fetch_prompt_templates(
        self, *,
        id: Optional[UUID] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        group: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        added_by: Optional[str] = None,
        template: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        exact_match_fields: Optional[List[str]] = None,
    ):
        pass

    def get_prompt_template_names(self, *, ):
        pass

    def get_dataset_names(self, *, substring: Optional[str] = None) -> list[str]:
        pass

    def fetch_prompts(
        self, *,
        id: Optional[UUID] = None,
        dataset_name: Optional[str] = None,
        value: Optional[str] = None,
        data_type: Optional[str] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        group: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        added_by: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        exact_match_fields: Optional[List[str]] = None,
    ):
        pass

    def push(self, *,
        data: PromptDataset | PromptTemplate | list[PromptTemplate],
        update_existing_data=False, # method fails if there are records with matching IDs in the DB unless this is set to True
    ):
        pass
