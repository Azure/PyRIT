from typing import List, Tuple

from pyrit.models import  (
    QuestionAnsweringDataset, 
    QuestionAnsweringEntry,
)

def get_question_prompt_pairs(
        dataset: QuestionAnsweringDataset,
    ):
    """
    Generates a list of question and answer pairs from the dataset.
    """
    QAList:List[Tuple[QuestionAnsweringEntry, str]] = []
    for question in dataset.questions:
        prompt = construct_evaluation_prompt(entry=question)
        QAList.append((question, prompt))
    return QAList

def construct_evaluation_prompt(entry: QuestionAnsweringEntry) -> str:
    available_choices = "\n".join(
        f"index={c.index}, value={c.text}" for c in entry.choices
    )
    prompt = (
        f"Question:\n"
        f"{entry.question}\n\n"
        f"Choices:\n"
        f"{available_choices}"
    )
    return prompt