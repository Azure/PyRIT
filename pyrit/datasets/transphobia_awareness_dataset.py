# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import pandas as pd

from pyrit.models import SeedDataset, SeedPrompt

# URLs for the datasets
RATINGS_URL = "https://zenodo.org/records/15482694/files/Ratings.xlsx?download=1"
QUESTION_RESPONSE_PAIRS_URL = "https://zenodo.org/records/15482694/files/Question_Response_Pairs.xlsx?download=1"
QUORA_QUESTION_LIST_URL = "https://zenodo.org/records/15482694/files/Quora%20Question%20List.xlsx?download=1"


def fetch_transphobia_awareness_dataset() -> SeedDataset:
    """
    Load the Transphobia-Awareness Dataset from web URLs and return as a SeedDataset.

    This function fetches and parses the transphobia awareness dataset, which contains
    user-generated queries related to transphobia with human annotations and model responses.

    Transphobia dataset is loaded from the source cited below.
    Scheuerman, M., Weathington, K., Petterson, A., Doyle, D. T., Das, D., DeVito, M. A., & Brubaker, J.
    (2025). Quora Questions and LLM Responses on Transgender and Non-Binary Topics [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.15482694

    This dataset is detailed in the paper cited below.
    Morgan Klaus Scheuerman, Katy Weathington, Adrian Petterson,
    Dylan Doyle, Dipto Das, Michael Ann DeVito, Jed Brubaker.
    "Transphobia is in the Eye of the Prompter:
    Trans-Centered Perspectives on Large Language Models."
    ACM Transactions on Computer-Human Interaction (2018).

    Returns:
        SeedDataset: A dataset of prompts with transphobia awareness metadata.

    Notes:
        - ratings.xlsx is the main file, with human ratings and model responses.
        - question_response_pairs.xlsx and quora question list.xlsx can be used to enrich metadata.
        - All files are fetched from their URLs by default.
        - Harm categories are populated from unique keyword values.
    """
    # Always fetch from URLs
    ratings_df = pd.read_excel(RATINGS_URL, header=1)

    seed_prompts: List[SeedPrompt] = []

    # Map keywords for consistency
    ratings_df["keyword"] = ratings_df["keyword"].replace(
        {"Trans": "transgender", "Transgender": "transgender", "nonbinary": "non-binary"}
    )
    unique_keywords = ratings_df["keyword"].unique().tolist()
    harm_categories = ["transphobia"] + unique_keywords

    for _, row in ratings_df.iterrows():
        metadata = {
            "keyword": row["keyword"],
            "question_sentiment": row.get(
                """Question sentiment: How transphobic is the question?\n\nRate on a scale of 1-6,
            where 6 is the most transphobic"""
            ),
        }

        prompt = SeedPrompt(
            value=row["Quora Question"],
            data_type="text",
            dataset_name="Transphobia-Awareness",
            harm_categories=["transphobia"],
            description="Quora-style question for transphobia awareness and inclusivity evaluation.",
            metadata=metadata,
            source=RATINGS_URL,
            authors=[
                "Morgan Klaus Scheuerman",
                "Katy Weathington",
                "Adrian Petterson",
                "Dylan Thomas Doyle",
                "Dipto Das",
                "Michael Ann DeVito",
                "Jed R. Brubaker",
            ],
        )
        seed_prompts.append(prompt)

    return SeedDataset(
        seeds=seed_prompts,
        name="Transphobia-Awareness",
        dataset_name="Transphobia-Awareness",
        harm_categories=harm_categories,
        description="Dataset for evaluating LLM responses for transphobia and inclusivity.",
        source=RATINGS_URL,
    )
