# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions import MissingPromptPlaceholderException
from pyrit.models import PromptRequestResponse, PromptRequestPiece, Score, SeedPromptDataset, SeedPrompt
from pyrit.prompt_converter import ConverterResult, FuzzerExpandConverter, FuzzerConverter, FuzzerShortenConverter
from pyrit.orchestrator import FuzzerOrchestrator
from pyrit.orchestrator.fuzzer_orchestrator import PromptNode
from pyrit.memory import CentralMemory
from pyrit.score import Scorer
from tests.mocks import MockPromptTarget
from pyrit.memory.memory_interface import MemoryInterface
import pathlib
import pytest
from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def mock_central_memory_instance(memory_interface):
    """Fixture to mock CentralMemory.get_memory_instance"""
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface) as duck_db_memory:
        yield duck_db_memory


@pytest.fixture
def scoring_target(mock_central_memory_instance) -> MockPromptTarget:
    return MockPromptTarget()


@pytest.fixture
def simple_prompts() -> list[SeedPrompt]:
    """sample prompts"""
    prompts = SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal.prompt")
    return prompts.prompts


@pytest.fixture
def simple_templateconverter(scoring_target) -> list[FuzzerConverter]:
    """template converter"""
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    prompt_expand_converter = FuzzerExpandConverter(converter_target=scoring_target)
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    return template_converters


@pytest.fixture
def simple_prompt_templates():
    """sample prompt templates that can be given as input"""
    prompt_template1 = SeedPrompt.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )
    prompt_template2 = SeedPrompt.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aim.yaml"
    )
    prompt_template3 = SeedPrompt.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aligned.yaml"
    )
    prompt_template4 = SeedPrompt.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "axies.yaml"
    )

    prompt_templates = [
        prompt_template1.value,
        prompt_template2.value,
        prompt_template3.value,
        prompt_template4.value,
    ]

    return prompt_templates


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 6)))
@pytest.mark.parametrize("success_pattern", ["1_per_round", "1_every_other_round"])
async def test_execute_fuzzer(
    rounds: int, success_pattern: str, simple_prompts: list, simple_prompt_templates: list, scoring_target
):

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    prompt_expand_converter = FuzzerExpandConverter(converter_target=scoring_target)
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=scoring_target,
        template_converters=template_converters,
        scoring_target=MagicMock(),
        target_jailbreak_goal_count=rounds,
    )
    prompt_node = fuzzer_orchestrator._initial_prompt_nodes
    fuzzer_orchestrator._scorer = MagicMock()

    true_score = Score(
        score_value="True",
        score_value_description="",
        score_type="true_false",
        score_category="",
        score_rationale="",
        score_metadata="",
        prompt_request_response_id="",
    )
    false_score = Score(
        score_value="False",
        score_value_description="",
        score_type="true_false",
        score_category="",
        score_rationale="",
        score_metadata="",
        prompt_request_response_id="",
    )
    prompt_target_response = [
        PromptRequestResponse(
            request_pieces=[PromptRequestPiece(original_value=prompt, converted_value=prompt, role="assistant")]
        )
        for prompt in simple_prompts
    ]
    with patch.object(fuzzer_orchestrator, "_select_template_with_mcts") as mock_get_seed:
        with patch.object(fuzzer_orchestrator, "_apply_template_converter") as mock_apply_template_converter:
            with patch.object(fuzzer_orchestrator, "_update"):
                mock_get_seed.return_value = prompt_node[0]
                mock_apply_template_converter.return_value = prompt_node[0].template
                fuzzer_orchestrator._prompt_normalizer = AsyncMock()
                fuzzer_orchestrator._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
                    return_value=prompt_target_response
                )
                fuzzer_orchestrator._scorer = AsyncMock()
                fuzzer_orchestrator._scorer.score_prompts_batch_async = AsyncMock()
                if success_pattern == "1_per_round":
                    fuzzer_orchestrator._scorer.score_prompts_batch_async.return_value = [false_score] * (
                        len(simple_prompts) - 1
                    ) + [true_score]
                elif success_pattern == "1_every_other_round":
                    fuzzer_orchestrator._scorer.score_prompts_batch_async.side_effect = [
                        [false_score] * len(simple_prompts),
                        [false_score] * (len(simple_prompts) - 1) + [true_score],
                    ] * rounds
                else:
                    raise ValueError("Invalid success_pattern.")

                result = await fuzzer_orchestrator.execute_fuzzer()

                assert result.success
                assert fuzzer_orchestrator._total_jailbreak_count == rounds
                assert result.description == "Maximum number of jailbreaks reached."
                assert len(result.templates) == rounds
                assert len(prompt_node[0].children) == rounds


def test_prompt_templates(simple_prompts: list, simple_templateconverter: list[FuzzerConverter], scoring_target):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=[],
            prompt_target=scoring_target,
            template_converters=simple_templateconverter,
            scoring_target=MagicMock(),
        )
    assert e.match("The initial set of prompt templates cannot be empty.")


def test_invalid_batchsize(
    simple_prompts: list, simple_prompt_templates: list, simple_templateconverter: list[FuzzerConverter], scoring_target
):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=simple_prompt_templates,
            prompt_target=scoring_target,
            template_converters=simple_templateconverter,
            scoring_target=MagicMock(),
            batch_size=0,
        )
    assert e.match("Batch size must be at least 1.")


def test_prompts(simple_prompt_templates: list, scoring_target):
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    prompt_expand_converter = FuzzerExpandConverter(converter_target=scoring_target)
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=[],
            prompt_templates=simple_prompt_templates,
            prompt_target=scoring_target,
            template_converters=template_converters,
            scoring_target=MagicMock(),
        )
    assert e.match("The initial prompts cannot be empty.")


def test_template_converter(simple_prompts: list, simple_prompt_templates: list, scoring_target):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=simple_prompt_templates,
            prompt_target=scoring_target,
            template_converters=[],
            scoring_target=MagicMock(),
        )
    assert e.match("Template converters cannot be empty.")


@pytest.mark.asyncio
async def test_max_query(simple_prompts: list, simple_prompt_templates: list, scoring_target):
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    prompt_expand_converter = FuzzerExpandConverter(converter_target=scoring_target)
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=scoring_target,
        template_converters=template_converters,
        scoring_target=MagicMock(),
    )

    assert fuzzer_orchestrator._max_query_limit == 80

    fuzzer_orchestrator._total_target_query_count = 79
    result = await fuzzer_orchestrator.execute_fuzzer()
    assert result.success is False
    assert result.description == "Query limit reached."


@pytest.mark.asyncio
async def test_apply_template_converter(simple_prompts: list, simple_prompt_templates: list, scoring_target):
    prompt_template = SeedPrompt.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )

    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=scoring_target,
        template_converters=[
            FuzzerShortenConverter(converter_target=scoring_target),
            FuzzerExpandConverter(converter_target=scoring_target),
        ],
        scoring_target=MagicMock(),
    )

    def get_mocked_converter(_):
        return fuzzer_orchestrator._template_converters[0]

    with patch("random.choice", get_mocked_converter):
        new_template = "new template {{ prompt }}"
        fuzzer_orchestrator._template_converters[0].convert_async = AsyncMock(  # type: ignore
            return_value=ConverterResult(output_text=new_template, output_type="text")
        )
        generated_template = await fuzzer_orchestrator._apply_template_converter(
            template=prompt_template, other_templates=[prompt_template.value]
        )
        TEMPLATE_PLACEHOLDER = "{{ prompt }}"
        assert TEMPLATE_PLACEHOLDER in generated_template


@pytest.mark.asyncio
async def test_apply_template_converter_empty_placeholder(
    simple_prompts: list[str], simple_prompt_templates: list, scoring_target
):
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=scoring_target,
        template_converters=[prompt_shorten_converter],
        scoring_target=MagicMock(),
    )

    prompt_shorten_converter.convert_async = AsyncMock(  # type: ignore
        return_value=ConverterResult(output_text="new template without prompt placeholder", output_type="text")
    )
    with pytest.raises(MissingPromptPlaceholderException) as e:
        await fuzzer_orchestrator._apply_template_converter(
            template=simple_prompt_templates[0], other_templates=[simple_prompt_templates[1]]
        )
        prompt_shorten_converter.convert_async.assert_called_once()
    assert e.match("Prompt placeholder is empty.")


@pytest.mark.asyncio
async def test_best_UCT(simple_prompts: list, simple_prompt_templates: list, scoring_target):
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    prompt_expand_converter = FuzzerExpandConverter(converter_target=scoring_target)
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=scoring_target,
        template_converters=template_converters,
        scoring_target=MagicMock(),
    )
    fuzzer_orchestrator._initial_prompt_nodes[0].visited_num = 2

    UCT_scores = [0.33, 2.0, 3.0, 0.0, 0.0]
    prompt_nodes = fuzzer_orchestrator._initial_prompt_nodes
    prompt_nodes[0].rewards = 1
    prompt_nodes[1].rewards = 2
    prompt_nodes[2].rewards = 3
    for index, node in enumerate(prompt_nodes):
        fuzzer_orchestrator._step = 1
        UCT_score_func = fuzzer_orchestrator._best_UCT_score()
        UCT_score = UCT_score_func(node)
        assert float(round(UCT_score, 2)) == UCT_scores[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("probability", [0, 0.5])
async def test_select(simple_prompts: list, probability: int, simple_prompt_templates: list, scoring_target):
    # set the children of each parent
    prompt_shorten_converter = FuzzerShortenConverter(converter_target=scoring_target)
    prompt_expand_converter = FuzzerExpandConverter(converter_target=scoring_target)
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=[simple_prompt_templates[0], simple_prompt_templates[1]],
        prompt_target=scoring_target,
        template_converters=template_converters,
        scoring_target=MagicMock(),
        frequency_weight=0.5,
        reward_penalty=0.1,
        minimum_reward=0.2,
        non_leaf_node_probability=0.1,
        batch_size=10,
    )

    new_node1 = PromptNode(simple_prompt_templates[2], parent=fuzzer_orchestrator._initial_prompt_nodes[1])
    PromptNode(simple_prompt_templates[3], parent=new_node1)

    fuzzer_orchestrator._step = 2
    prompt_node = fuzzer_orchestrator._initial_prompt_nodes
    prompt_node[0].rewards = 1
    prompt_node[1].rewards = 2

    def get_mocked_random_number():
        return probability

    with patch("numpy.random.rand", get_mocked_random_number):
        if probability == 0:
            fuzzer_orchestrator._select_template_with_mcts()
            path_mcts = fuzzer_orchestrator._mcts_selected_path
            for node in path_mcts:
                assert node.parent is None
                assert len(path_mcts) == 1

        if probability == 0.5:
            fuzzer_orchestrator._select_template_with_mcts()
            path_mcts = fuzzer_orchestrator._mcts_selected_path
            assert path_mcts[0].parent is None
            assert path_mcts[2].parent == path_mcts[0].children[0]
            assert path_mcts[2].children == []
            assert len(path_mcts) == 3
