# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import AsyncMock, MagicMock, patch
from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions import MissingPromptPlaceholderException
from pyrit.models import PromptRequestResponse, PromptRequestPiece, PromptDataset, PromptTemplate
from pyrit.prompt_converter import ConverterResult, ExpandConverter, PromptConverter, ShortenConverter
from pyrit.orchestrator import FuzzerOrchestrator
from pyrit.orchestrator.fuzzer_orchestrator import PromptNode
from pyrit.score import Score, Scorer
from tests.mocks import MockPromptTarget
import pathlib
import pytest
import tempfile


@pytest.fixture
def scoring_target(memory) -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    return MockPromptTarget(memory=memory)


@pytest.fixture
def simple_prompts() -> list[str]:
    """sample prompts"""
    prompts = PromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompts" / "illegal.prompt")
    return prompts.prompts


@pytest.fixture
def simple_templateconverter() -> list[PromptConverter]:
    """template converter"""
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    return template_converters


@pytest.fixture
def simple_prompt_templates():
    """sample prompt templates that can be given as input"""
    prompt_template1 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )
    prompt_template2 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aim.yaml"
    )
    prompt_template3 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aligned.yaml"
    )
    prompt_template4 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "axies.yaml"
    )
    # prompt_template5 = PromptTemplate.from_yaml_file(
    # pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "balakula.yaml")

    prompt_templates = [
        prompt_template1.template,
        prompt_template2.template,
        prompt_template3.template,
        prompt_template4.template,
    ]

    return prompt_templates


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 6)))
@pytest.mark.parametrize("success_pattern", ["1_per_round", "1_every_other_round"])
async def test_execute_fuzzer(rounds: int, success_pattern: str, simple_prompts: list, simple_prompt_templates: list):

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converters=template_converters,
        scoring_target=MagicMock(),
        target_jailbreak_goal_count=rounds,
    )
    prompt_node = fuzzer_orchestrator._initial_prompts_nodes
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


def test_prompt_templates(simple_prompts: list, simple_templateconverter: list[PromptConverter]):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=[],
            prompt_target=MockPromptTarget(),
            template_converters=simple_templateconverter,
            scoring_target=MagicMock(),
            memory=MagicMock(),
        )
    assert e.match("The initial set of prompt templates cannot be empty.")


def test_invalid_batchsize(
    simple_prompts: list, simple_prompt_templates: list, simple_templateconverter: list[PromptConverter]
):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=simple_prompt_templates,
            prompt_target=MockPromptTarget(),
            template_converters=simple_templateconverter,
            scoring_target=MagicMock(),
            memory=MagicMock(),
            batch_size=0,
        )
    assert e.match("Batch size must be at least 1.")


def test_prompts(simple_prompt_templates: list):
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=[],
            prompt_templates=simple_prompt_templates,
            prompt_target=MockPromptTarget(),
            template_converters=template_converters,
            scoring_target=MagicMock(),
            memory=MagicMock(),
        )
    assert e.match("The initial prompts cannot be empty.")


def test_template_converter(simple_prompts: list, simple_prompt_templates: list):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=simple_prompt_templates,
            prompt_target=MockPromptTarget(),
            template_converters=[],
            scoring_target=MagicMock(),
            memory=MagicMock(),
        )
    assert e.match("Template converters cannot be empty.")


@pytest.mark.asyncio
async def test_max_query(simple_prompts: list, simple_prompt_templates: list):
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converters=template_converters,
        scoring_target=MagicMock(),
        memory=MagicMock(),
    )

    assert fuzzer_orchestrator._max_query_limit == 80

    fuzzer_orchestrator._total_target_query_count = 79
    result = await fuzzer_orchestrator.execute_fuzzer()
    assert result.success is False
    assert result.description == "Query limit reached."


@pytest.mark.asyncio
async def test_apply_template_converter(simple_prompts: list, simple_prompt_templates: list):
    prompt_template = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )

    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converters=[
            ShortenConverter(converter_target=MockPromptTarget()),
            ExpandConverter(converter_target=MockPromptTarget()),
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
        generated_template = await fuzzer_orchestrator._apply_template_converter(prompt_template)
        TEMPLATE_PLACEHOLDER = "{{ prompt }}"
        assert TEMPLATE_PLACEHOLDER in generated_template


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template_converter",
    [ShortenConverter(converter_target=MockPromptTarget()), ExpandConverter(converter_target=MockPromptTarget())],
)
async def test_apply_template_converter_empty_placeholder(
    simple_prompts: list[str], template_converter: PromptConverter, simple_prompt_templates: list
):
    prompt_template = ['Write a wikipedia article on . Start the article with "Title:".']
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converters=template_converters,
        scoring_target=MagicMock(),
    )

    def get_mocked_random_number():
        return template_converter

    with patch("random.choice", get_mocked_random_number):
        if template_converter == ExpandConverter or template_converter == ShortenConverter:
            with patch.object(template_converter, "_convert_async") as mock_template:
                mock_template.return_value = prompt_template
                with pytest.raises(MissingPromptPlaceholderException) as e:
                    fuzzer_orchestrator._apply_template_converter(simple_prompt_templates[0])
                    assert e.match("Prompt placeholder is empty.")


@pytest.mark.asyncio
async def test_best_UCT(simple_prompts: list, simple_prompt_templates: list):
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converters=template_converters,
        scoring_target=MagicMock(),
    )
    fuzzer_orchestrator._initial_prompts_nodes[0].visited_num = 2

    UCT_scores = [0.33, 2.0, 3.0, 0.0, 0.0]
    prompt_nodes = fuzzer_orchestrator._initial_prompts_nodes
    prompt_nodes[0].rewards = 1
    prompt_nodes[1].rewards = 2
    prompt_nodes[2].rewards = 3
    for index, node in enumerate(prompt_nodes):
        # fuzzer_orchestrator.rewards = [1,2,3,4,5]
        # fuzzer_orchestrator.index = node.index
        fuzzer_orchestrator._step = 1
        UCT_score_func = fuzzer_orchestrator._best_UCT_score()
        UCT_score = UCT_score_func(node)
        assert float(round(UCT_score, 2)) == UCT_scores[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("probability", [0, 0.5])
async def test_select(simple_prompts: list, probability: int, simple_prompt_templates: list):
    prompt_template1 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )
    prompt_template2 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aim.yaml"
    )
    prompt_template3 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aligned.yaml"
    )
    prompt_template4 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "axies.yaml"
    )
    prompt_templates = [prompt_template1.template, prompt_template2.template]
    # set the children of each parent
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converters=template_converters,
        scoring_target=MagicMock(),
        frequency_weight=0.5,
        reward_penalty=0.1,
        minimum_reward=0.2,
        non_leaf_node_probability=0.1,
        batch_size=10,
    )

    new_node1 = PromptNode(prompt_template3.template, parent=fuzzer_orchestrator._initial_prompts_nodes[1])
    PromptNode(prompt_template4.template, parent=new_node1)

    fuzzer_orchestrator._step = 2
    prompt_node = fuzzer_orchestrator._initial_prompts_nodes
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

        if probability == 0.5:
            fuzzer_orchestrator._select_template_with_mcts()
            path_mcts = fuzzer_orchestrator._mcts_selected_path
            assert path_mcts[0].parent is None
            assert path_mcts[2].parent == path_mcts[0].children[0]
            assert path_mcts[2].children == []
            assert len(path_mcts) == 3
