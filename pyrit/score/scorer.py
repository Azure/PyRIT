# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import json
from typing import Callable, List, Tuple

import pydantic
import tenacity
from openai import OpenAI

from pyrit.common.logger import logger
from pyrit.models import ExamAnswers, PromptDataset, ScoreAnswers, ScoringResults

API_KEY = ""


class ChatCompletion(pydantic.BaseModel):
    finish_reason: str
    content: str


@tenacity.retry(wait=tenacity.wait_fixed(1), stop=tenacity.stop_after_delay(8))
def _create_gpt_chat_completion(api_key: str, system_content: str, user_content: str) -> ChatCompletion:
    client = OpenAI(api_key=api_key)
    openai_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )
    completion = ChatCompletion(
        finish_reason=openai_completion.choices[0].finish_reason,
        content=openai_completion.choices[0].message.content,
    )
    return completion


def classify_content(
    content: str,
    classification_prompt: str,
    classification_enum: Callable[[str], enum.Enum],
    invalid_okay: bool = False,
) -> Tuple[enum.Enum, str]:
    """Classify the content using the classification_prompt and classification_enum.

    Returns:
        Tuple in the (classification, explanation) format.
    """

    completion = _create_gpt_chat_completion(
        api_key=API_KEY, system_content=classification_prompt, user_content=content
    )
    predicted_classification_letter = completion.content.splitlines()[0]
    predicted_explanation = "".join(completion.content.splitlines()[1:])
    class_label = classification_enum(predicted_classification_letter)
    return class_label, predicted_explanation


def remove_xml_from_text(text: str) -> str:
    response_xml_tag = "</Response>"
    before_tag_text, _ = text.split(response_xml_tag)
    return before_tag_text


def extract_json_keys_from_text(text: str) -> dict:
    json_response = json.loads(text)
    return json_response


def coerce_response_to_dict(text_response: str) -> dict:
    """Take a response from GPT and remove extra characters to coherse into having a standard dict format

    With the current metaprompt, the output is known to contain the "</Response>" tag, so this removes is and returns
    a valid dict
    """

    # Remove everything until the first curly bracket
    first_curly_bracket_idx = text_response.find("{")
    text_response = text_response[first_curly_bracket_idx:]

    # Trimp the </Response> from output
    known_extra_tags = ["</Response>"]
    for extra_tag in known_extra_tags:
        if extra_tag in text_response:
            before_tag_text, _ = text_response.split(extra_tag)
            text_response = before_tag_text

    # Remove special non-ascii charactes from string. Sometimes the response contains zero-width special characters
    try:
        text_response = text_response.encode("ascii", "ignore").decode()
        response_dict = json.loads(text_response)
    except json.JSONDecodeError:
        logger.exception(
            f"FA1F9C3B-60A7-4FB9-8E41-E769FA80578C. Error while decoding response." f'[response="{text_response}"]'
        )
        raise

    expected_keys = ["final_response", "internal_reasoning", "did_block_request"]
    try:
        assert set(response_dict.keys()) == set(expected_keys)
    except AssertionError:
        logger.debug(
            f"617A7D00-08DC-4F0F-A5D4-6B0984278DE8. Seems like one of the responses does not have the "
            f'expected JSON format. [response="{text_response}"]'
        )
    return response_dict


def _normalize_choices(text: str) -> List[str]:
    """Takes a string and ensures that a list of the single-character choices
    are returned.

    For example:
        > normalize_choices('A, ')
        ['A']
        > normalize_choices('ABC')
        ['A', 'B', 'C']
        > normalize_choices('A,B')
        ['A', 'B']

    """
    tmp = text.replace(" ", "")
    tmp = tmp.replace(",", "")
    return [x for x in tmp]


def _print_results_to_console(question: str, user_answer: str, correct_answer: str, is_correct: bool):
    output_string = f"""
    Question:
    {question}

    User answer:
    {user_answer}

    Correct answer:
    {correct_answer}

    Is correct?:
    {is_correct}
    """
    print(output_string)


class ScoringEngine:
    @staticmethod
    def score_exam_answers(
        questions: PromptDataset,
        answers_to_grade: ExamAnswers,
        correct_answers: ScoreAnswers,
        verbose: bool = False,
    ) -> ScoringResults:
        assert len(answers_to_grade.answer) == len(correct_answers.answers) == len(questions.prompts)
        score_card = ScoringResults(failed=0, passed=0, passed_with_partial_credit=0, questions_count=0)

        for user_answer, correct_answer, prompt in zip(
            answers_to_grade.answer, correct_answers.answers, questions.prompts
        ):
            score_card.questions_count += 1

            user_answer_key_set = set(_normalize_choices(user_answer.answer))
            correct_answer_key_set = set(_normalize_choices(correct_answer))

            # Case without partial credit
            if user_answer_key_set == correct_answer_key_set:
                score_card.passed += 1
            else:
                score_card.failed += 1
            # Case with partial credit
            num_correct_answer = len(correct_answer_key_set)
            num_overlapped_answers = len(correct_answer_key_set.intersection(user_answer_key_set))
            score_card.passed_with_partial_credit += num_overlapped_answers / num_correct_answer

            if num_overlapped_answers:
                if verbose:
                    _print_results_to_console(
                        question=prompt,
                        user_answer=user_answer.answer,
                        correct_answer=correct_answer,
                        is_correct=user_answer_key_set == correct_answer_key_set,
                    )

        return score_card
