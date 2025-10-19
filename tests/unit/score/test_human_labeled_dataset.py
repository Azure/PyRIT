# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import pytest

from pyrit.common.path import SCORER_EVALS_HARM_PATH
from pyrit.models import MessagePiece, Message
from pyrit.score import (
    HarmHumanLabeledEntry,
    HumanLabeledDataset,
    MetricsType,
    ObjectiveHumanLabeledEntry,
)


@pytest.fixture
def sample_responses():
    return [
        Message(
            message_pieces=[
                MessagePiece(role="assistant", original_value="test response", original_value_data_type="text")
            ]
        )
    ]


def test_harm_human_labeled_entry_valid(sample_responses):
    entry = HarmHumanLabeledEntry(sample_responses, [0.0, 0.5, 1.0], "hate_speech")
    assert entry.harm_category == "hate_speech"
    assert entry.human_scores == [0.0, 0.5, 1.0]


def test_harm_human_labeled_entry_invalid_score(sample_responses):
    with pytest.raises(ValueError):
        HarmHumanLabeledEntry(sample_responses, [-0.1, 1.1], "hate_speech")


def test_objective_human_labeled_entry(sample_responses):
    entry = ObjectiveHumanLabeledEntry(sample_responses, [True, False], "how to make a Molotov cocktail?")
    assert entry.objective == "how to make a Molotov cocktail?"
    assert entry.human_scores == [True, False]


def test_human_labeled_dataset_init_harm_type(sample_responses):
    entry = HarmHumanLabeledEntry(sample_responses, [0.1, 0.2], "hate_speech")
    dataset = HumanLabeledDataset(name="hate_speech", entries=[entry], metrics_type=MetricsType.HARM)
    assert dataset.name == "hate_speech"
    assert dataset.metrics_type == MetricsType.HARM
    assert len(dataset.entries) == 1
    assert isinstance(dataset.entries[0], HarmHumanLabeledEntry)


def test_human_labeled_dataset_init_objective_type(sample_responses):
    entry = ObjectiveHumanLabeledEntry(sample_responses, [True, False], "objective")
    dataset = HumanLabeledDataset(name="sample_objective", entries=[entry], metrics_type=MetricsType.OBJECTIVE)
    assert dataset.name == "sample_objective"
    assert dataset.metrics_type == MetricsType.OBJECTIVE
    assert len(dataset.entries) == 1
    assert isinstance(dataset.entries[0], ObjectiveHumanLabeledEntry)


def test_human_labeled_dataset_init_empty_args(sample_responses):
    entry = ObjectiveHumanLabeledEntry(sample_responses, [True, False], "objective")
    with pytest.raises(ValueError):
        HumanLabeledDataset(name="", entries=[entry], metrics_type=MetricsType.OBJECTIVE)


def test_human_labeled_dataset_init_mismatch_metrics_type_error(sample_responses):
    entry = ObjectiveHumanLabeledEntry(sample_responses, [True, False], "objective")
    with pytest.raises(ValueError):
        HumanLabeledDataset(name="harm", entries=[entry], metrics_type=MetricsType.HARM)


def test_human_labeled_dataset_init_multiple_harm_categories_warns(sample_responses, caplog):
    entry1 = HarmHumanLabeledEntry(sample_responses, [0.1], "hate_speech")
    entry2 = HarmHumanLabeledEntry(sample_responses, [0.2], "violence")
    with caplog.at_level(logging.WARNING):
        HumanLabeledDataset(name="mixed", entries=[entry1, entry2], metrics_type=MetricsType.HARM)
        assert any("All entries in a harm dataset should have the same harm category" in m for m in caplog.messages)


def test_human_labeled_dataset_add_entry(sample_responses):
    entry = HarmHumanLabeledEntry(sample_responses, [0.2, 0.8], "hate_speech")
    dataset = HumanLabeledDataset(name="hate_speech", entries=[], metrics_type=MetricsType.HARM)
    dataset.add_entry(entry)
    assert len(dataset.entries) == 1
    assert isinstance(dataset.entries[0], HarmHumanLabeledEntry)


def test_human_labeled_dataset_add_entry_warns(sample_responses, caplog):
    entry = HarmHumanLabeledEntry(sample_responses, [0.1, 0.2], "hate_speech")
    entry2 = HarmHumanLabeledEntry(sample_responses, [0.1, 0.2], "violence")
    dataset = HumanLabeledDataset(name="hate_speech", entries=[entry], metrics_type=MetricsType.HARM)
    with caplog.at_level(logging.WARNING):
        dataset.add_entry(entry2)
        assert any("All entries in a harm dataset should have the same harm category" in m for m in caplog.messages)


def test_human_labeled_dataset_add_entries(sample_responses):
    entry1 = HarmHumanLabeledEntry(sample_responses, [0.1, 0.2], "hate_speech")
    entry2 = HarmHumanLabeledEntry(sample_responses, [0.3, 0.4], "hate_speech")
    dataset = HumanLabeledDataset(name="hate_speech", entries=[], metrics_type=MetricsType.HARM)
    dataset.add_entries([entry1, entry2])
    assert len(dataset.entries) == 2
    assert all(isinstance(e, HarmHumanLabeledEntry) for e in dataset.entries)


def test_human_labeled_dataset_validate_entry_type_error(sample_responses):
    entry = ObjectiveHumanLabeledEntry(sample_responses, [True, False], "objective")
    dataset = HumanLabeledDataset(name="hate_speech", entries=[], metrics_type=MetricsType.HARM)
    with pytest.raises(ValueError):
        dataset.add_entry(entry)


def test_human_labeled_dataset_validate_columns_missing_column():
    import pandas as pd

    df = pd.DataFrame(
        {
            "assistant_response": ["a"],
            "label1": [0.1],
            # "label2" is missing
            "harm_category": ["hate_speech"],
        }
    )
    with pytest.raises(ValueError):
        HumanLabeledDataset._validate_columns(
            eval_df=df,
            human_label_col_names=["label1", "label2"],
            assistant_response_col_name="assistant_response",
            objective_or_harm_col_name="harm_category",
        )


def test_human_labeled_dataset_validate_columns_nan():
    import pandas as pd

    df = pd.DataFrame(
        {"assistant_response": ["a"], "label1": [None], "label2": [0.2], "harm_category": ["hate_speech"]}
    )
    with pytest.raises(ValueError):
        HumanLabeledDataset._validate_columns(
            eval_df=df,
            human_label_col_names=["label1", "label2"],
            assistant_response_col_name="assistant_response",
            objective_or_harm_col_name="harm_category",
        )


def test_validate_fields_valid():
    HumanLabeledDataset._validate_fields(
        response_to_score="Some response",
        human_scores=[0.5, 1, 0],
        objective_or_harm="some_objective",
        data_type="text",
    )


def test_validate_fields_empty_response():
    with pytest.raises(ValueError):
        HumanLabeledDataset._validate_fields(
            response_to_score="   ", human_scores=[0.5, 1], objective_or_harm="some_objective", data_type="text"
        )


def test_validate_fields_invalid_human_scores():
    with pytest.raises(ValueError):
        HumanLabeledDataset._validate_fields(
            response_to_score="Some response",
            human_scores=[0.5, "bad"],
            objective_or_harm="some_objective",
            data_type="text",
        )


def test_validate_fields_missing_objective_or_harm():
    with pytest.raises(ValueError):
        HumanLabeledDataset._validate_fields(
            response_to_score="Some response", human_scores=[0.5, 1], objective_or_harm="", data_type="text"
        )


def test_validate_fields_invalid_data_type():
    with pytest.raises(ValueError):
        HumanLabeledDataset._validate_fields(
            response_to_score="Some response",
            human_scores=[0.5, 1],
            objective_or_harm="some_objective",
            data_type="invalid_type",
        )


def test_construct_harm_entry(sample_responses):
    entry = HumanLabeledDataset._construct_harm_entry(
        request_responses=sample_responses, harm="hate_speech", human_scores=["0.1", 0.2]
    )
    assert entry.harm_category == "hate_speech"
    assert entry.human_scores == [0.1, 0.2]


def test_construct_objective_entry_bool_conversion(sample_responses):
    entry = HumanLabeledDataset._construct_objective_entry(
        request_responses=sample_responses, objective="obj", human_scores=["1", "0"]
    )
    assert entry.human_scores == [True, False]


def test_human_labeled_dataset_from_csv():
    csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/SAMPLE_hate_speech.csv"
    dataset = HumanLabeledDataset.from_csv(
        csv_path=csv_path,
        metrics_type=MetricsType.HARM,
        assistant_response_col_name="assistant_response",
        human_label_col_names=["human_score_1", "human_score_2", "human_score_3"],
        objective_or_harm_col_name="category",
    )
    assert isinstance(dataset, HumanLabeledDataset)
    assert dataset.metrics_type == MetricsType.HARM
    assert all(isinstance(e, HarmHumanLabeledEntry) for e in dataset.entries)
    assert dataset.name == "SAMPLE_hate_speech"


def test_human_labeled_dataset_from_csv_with_data_type_col(tmp_path):
    import pandas as pd

    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "assistant_response": ["response1"],
            "label1": [0.1],
            "label2": [0.2],
            "harm_category": ["hate_speech"],
            "data_type": ["text"],
        }
    )
    df.to_csv(csv_file, index=False)
    dataset = HumanLabeledDataset.from_csv(
        csv_path=str(csv_file),
        metrics_type=MetricsType.HARM,
        assistant_response_col_name="assistant_response",
        human_label_col_names=["label1", "label2"],
        objective_or_harm_col_name="harm_category",
        assistant_response_data_type_col_name="data_type",
    )
    assert isinstance(dataset, HumanLabeledDataset)
    assert dataset.entries[0].conversation[0].message_pieces[0].original_value_data_type == "text"
