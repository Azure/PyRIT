# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.score.markdown_injection import MarkdownInjectionScorer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "![sdf](www.google.com)",
        "![Drag Racing](Dragster.jpg)",
        "![][image_ref_a32ff4ads]",
    ],
)
async def test_md_injection_detected(text: str):
    scorer = MarkdownInjectionScorer()
    score = (await scorer.score_text_async(text))[0]
    assert score.get_value() is True


@pytest.mark.asyncio
async def test_md_injection_not_detected():
    classifier = MarkdownInjectionScorer()
    text = "this is a test"
    score = await classifier.score_text_async(text)
    assert score[0].get_value() is False
