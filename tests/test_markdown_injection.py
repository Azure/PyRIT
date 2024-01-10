# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.score.markdown_injection import MarkdownInjectionClassifier


@pytest.mark.parametrize(
    "text",
    [
        "![sdf](www.google.com)",
        "![Drag Racing](Dragster.jpg)",
        "![][image_ref_a32ff4ads]",
    ],
)
def test_md_injection_detected(text: str):
    classifier = MarkdownInjectionClassifier()
    score = classifier.score_text(text)
    assert score.score_value is True


def test_md_injection_not_detected():
    classifier = MarkdownInjectionClassifier()
    text = "this is a test"
    score = classifier.score_text(text)
    assert score.score_value is False
