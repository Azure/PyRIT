# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from dataclasses import dataclass
import logging
import random
import numpy as np
from typing import Optional
from uuid import uuid4
from colorama import Fore, Style
from pathlib import Path

from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.common.path import SCALES_PATH
from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score
from pyrit.score import SelfAskScaleScorer
from pyrit.score.float_scale_threshold_scorer import FloatScaleThresholdScorer
from pyrit.score.self_ask_category_scorer import ContentClassifierPaths
from pyrit.models import PromptTemplate, PromptDataset
from typing import Callable
from pyrit.prompt_converter import shorten_converter, expand_converter
from pyrit.prompt_target import PromptTarget, PromptChatTarget

from pyrit.exceptions import MissingPromptHolderException, pyrit_placeholder_retry

logger = logging.getLogger(__name__)


class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete


# todo remove variables and exports that are not necessary
class PromptNode:
    def __init__(
        self,
        template: str,
        parent: "PromptNode" = None,
    ):
        """ Class to maintain the tree information for each prompt template

        Args:

        template: Prompt template.

        """
    
        self._template: str = template

        self._parent: "PromptNode" = parent
        self._children: "list[PromptNode]" = []
        self._level: int = 0 if parent is None else parent._level + 1

        self._index: int = None
        self._visited_num = 0

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self._parent is not None:
            self._parent._children.append(self)


class FuzzerOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompts: PromptDataset,  # list[str]
        prompt_target: PromptTarget,
        prompt_templates: list[str],
        prompt_converters: Optional[list[PromptConverter]] = None,
        template_converter: list[PromptConverter],  # shorten/expand
        scoring_target: PromptChatTarget,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
        frequency_weight=0.5,
        reward_penalty=0.1,
        minimum_reward=0.2,
        non_leaf_node_probability=0.1,
        random_seed=None,
        batch_size: int = 10,
        jailbreak_goal = 1,
        query_limit: Optional[int] = None,
    ) -> None:
        """Creates an orchestrator that explores a variety of jailbreak options via fuzzing.

        Paper: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.

            Link: https://arxiv.org/pdf/2309.10253

            Authors: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

            GitHub: https://github.com/sherdencooper/GPTFuzz

        Args:

            prompts: The prompts will be the questions to the target.

            prompt_target: The target to send the prompts to.

            prompt_templates: List of all the jailbreak templates which will act as the seed pool.
            At each iteration, a seed will be selected using the MCTS-explore algorithm which will be sent to the
            shorten/expand prompt converter.

                          The converted template along with the prompt will be sent to the target.

            prompt_converters: The prompt_converters to use to convert the prompts before sending
            them to the prompt target.

            template_converter: The converter that will be applied on the jailbreak template that was
            selected by MCTS-explore.

                          The prompt converters will not be applied to the prompts.
                          Shorten/expand prompt converters.

            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory
            will be used.

            memory_labels: The labels to use for the memory. This is useful to identify the messages in the memory.

            verbose: Whether to print debug information.

            frequency_weight: constant that balances between the seed with high reward and the seed that is selected fewer times.

            reward_penalty: Reward penalty diminishes the reward for the current node and its ancestors when the path lengthens.

            minimum_reward: Minimal reward prevents the reward of the current node and its ancestors from being too small or negative.

            non_leaf_node_probability: parameter which decides the likelihood of selecting a non-leaf node.

            random_seed: Used to save the state of a random function.

            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.

            Jailbreak_goal: Maximum number of the jailbreaks after which the fuzzer will stop.

            Query_limit: Maximum number of times the fuzzer will run.

        """

        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target
        #self._achieved_objective = False
        self._prompts = prompts

        #self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        #self._prompt_target._memory = self._memory
        #self._prompt_target_conversation_id = str(uuid4())
        #self._memory = self._memory
        self._prompt_templates = prompt_templates
        self._template_converter = template_converter
        self._frequency_weight = frequency_weight
        self._reward_penalty = reward_penalty
        self._minimum_reward = minimum_reward
        self._non_leaf_node_probability = non_leaf_node_probability
        #Maximum number of the jailbreaks after which the fuzzer will stop
        self._jailbreak_goal = jailbreak_goal
        #Maximum number of times the fuzzer will run
        if query_limit:
            self._query_limit = query_limit
        else:
            self._query_limit = len(self._prompt_templates) * len(self._prompts) * 10
        self._current_query = 0
        self._current_jailbreak = 0
        self._batch_size = batch_size
        self._new_nodes = []
        global TEMPLATE_PLACEHOLDER 
        TEMPLATE_PLACEHOLDER = "{{ prompt }}"

        scorer_scale_path = Path(SCALES_PATH / "tree_of_attacks_with_pruning_jailbreak.yaml")
        scale_scorer = SelfAskScaleScorer(
            chat_target=scoring_target,
            scale_path=scorer_scale_path,
            memory=self._memory,
        )
        self._scorer = FloatScaleThresholdScorer(
            scorer=scale_scorer,
            threshold=0.8,
            memory=self._memory,
        )

        if not self._prompt_templates:
            raise ValueError("The initial set of prompt_templates cannot be empty.")
        if not self._prompts:
            raise ValueError("The initial prompts cannot be empty")
        if not self._template_converter:
            raise ValueError("Template converter cannot be empty")

        # if self._scorer.scorer_type != "true_false":
        #     raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {self._scorer.scorer_type}.")
        # # self._scorer = self._scorer

        if self._batch_size == 0:
            raise ValueError(f"Batch size must be at least 1.")

        # convert each template into a node and maintain the node information parent,child etc
        self._prompt_nodes: list[PromptNode] = [PromptNode(prompt) for prompt in prompt_templates]

        self._initial_prompts_nodes = self._prompt_nodes.copy()

        for i, prompt_node in enumerate(self._prompt_nodes):
            prompt_node.index = i

    async def execute_fuzzer(self) -> PromptRequestPiece:
        """
        Steps:
        1. Select a template - Send the prompt templates to the MCTS-explore to select a template at each iteration

        2. Apply template converter

        3. Append prompt with the selected template. For each selected template append all the prompts.

        4. Apply prompt converters if any and send it to target and get a response

        5. If jailbreak is successful then retain the template in template pool.

        6. Update the rewards for each of the node. update()

        Args:
            prompt_templates: A list of the initial jailbreak templates that needs to be sent to the MCTS algorithm.

        """
        # stopping criteria
        if self._current_query >= self._max_query:
            logger.info(f"Maximum query limit reached.")
            return FuzzerResult(success=False, nodes = self._new_nodes, prompt_target_conversation_id=conversation_id)
        
        elif self._current_jailbreak >= self._max_jailbreak:
            logger.info(f"Maximum number of jailbreaks reached.")
            return FuzzerResult(success=False, nodes = self._new_nodes, prompt_target_conversation_id=conversation_id)
        
        else:
            # 1. Select a seed from the list of the templates using the MCTS
            
            current_seed = await self._select(prompt_templates=self._prompt_templates)
            if TEMPLATE_PLACEHOLDER not in current_seed:
                raise MissingPromptHolderException(message="Prompt placeholder is empty.")

            # 2. Apply seed converter to the selected template.
            try:
                target_seed_obj = await self._apply_template_converter(current_seed._template)
            except:
                raise ValueError("Invalid template")

            target_template = PromptTemplate(target_seed_obj.output_text, parameters=["prompt"])

            # convert the target_template into a prompt_node to maintain the tree information
            target_template_node = PromptNode(template=target_seed_obj, parent=current_seed)

            # 3. Fill in prompts into the newly generated template.

            jailbreak_prompts = []
            for prompt in self._prompts:
                jailbreak_prompts.append(target_template.apply_custom_metaprompt_parameters(prompt=prompt))

            # 4. Apply prompt converter if any and send request to the target

            requests: list[NormalizerRequest] = []
            for jailbreak_prompt in jailbreak_prompts:
                requests.append(
                    self._create_normalizer_request(
                        prompt_text=jailbreak_prompt,
                        prompt_type="text",
                        converters=self._prompt_converters,
                    )
                )

            for request in requests:
                request.validate()

            responses: list[PromptRequestResponse]
            responses = await self._prompt_normalizer.send_prompt_batch_to_target_async(
                requests=requests,
                target=self._prompt_target,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
                batch_size=self._batch_size,
            )

            response_pieces = [response.request_pieces[0] for response in responses]
            conversation_id = [response.request_pieces[0].conversation_id for response in responses]
            # 5. Score the responses.

            scored_response = []
            for response in response_pieces:
                scored_response.append(self._scorer.score_async(response))

            batch_scored_response = await asyncio.gather(*scored_response)

            score_values = [score[0].get_value() for score in batch_scored_response]

            # 6. Update the rewards for each of the nodes.
            self._num_jailbreak = 0
            for index, score in enumerate(score_values):
                if score == True:
                    self._num_jailbreak += 1
                    conversation_id_jailbreak = conversation_id[index]
            num_executed_queries = len(score_values)

            self._current_jailbreak += self._num_jailbreak
            self._current_query += num_executed_queries

            if self._num_jailbreak > 0:  # successful jailbreak
                # append the template to the initial template list.
                self._prompt_nodes.append(target_template_node)
                self._new_nodes.append(target_template)
                return FuzzerResult(success = True, nodes = self._new_nodes, prompt_target_conversation_id=conversation_id_jailbreak )

            # update the rewards for the target_node
            self._update(target_template_node)

    def _select(self) -> PromptNode:
        """
            This method selects the template from the list of templates using the MCTS algorithm.
        """
        self._step = 0  # to keep track of the steps or the count
        self._last_choice_index = None
        # keeps track of the path that has been currently selected
        self._mcts_selected_path: list[PromptNode] = []  # type: ignore
        self._rewards = []
        self._step += 1
        # if the length of list of templates greater than rewards list (when new nodes added because of successful jailbreak) assign 0 as initial reward.
        if len(self._prompt_nodes) > len(self._rewards):
            self._rewards.extend([0 for _ in range(len(self._prompt_nodes) - len(self._rewards))])

        current = max(self._initial_prompts_nodes, key=self._best_UCT_score())  # initial path
        self._mcts_selected_path.append(current)

        # while node is not a leaf
        while len(current._children) > 0:
            if np.random.rand() < self._non_leaf_node_probability:
                break
            # compute the bestUCT score
            current = max(current._children, key=self._best_UCT_score())
            # append node to path
            self._mcts_selected_path.append(current)

        for prompt_node in self._mcts_selected_path:
            # keep track of number of visited nodes
            prompt_node._visited_num += 1

        self._last_choice_index = current.index
        # returns the best child
        return current

    def _best_UCT_score(self):
        """Function to compute the score for each seed. The highest-scoring seed will be selected as the next seed.

        This is an extension of the Monte carlo tree search (MCTS) algorithm which applies Upper Confidence Bound (UCB) to the node.

        The UCB function determines the confidence interval for each node and returns the highest value which will be selected as the next seed.
        """

        return lambda pn: self._rewards[pn.index] / (pn._visited_num + 1) + self._frequency_weight * np.sqrt(
            2 * np.log(self._step) / (pn._visited_num + 0.01)
        )  # self._frequency_weight - constant that balances between the seed with high reward and the seed that is selected fewer times.

    def _update(self, prompt_nodes: PromptNode):
        """
        Updates the reward of all the nodes in the last chosen path.
        """
        success_number = self._num_jailbreak  # check

        last_chosen_node = self._prompt_nodes[self._last_choice_index]
        for prompt_node in reversed(self._mcts_selected_path):
            # the output from the template converter will always be a single template so the length of prompt_nodes will be 1.
            reward = success_number / (len(self._prompts) * 1)  # len(prompt_nodes))
            self._rewards[prompt_node.index] += reward * max(
                self._minimum_reward, (1 - self._reward_penalty * last_chosen_node._level)
            )

    @pyrit_placeholder_retry
    async def _apply_template_converter(self, current_seed):
        """
        Asynchronously applies template converter.

        Retries the function if it raises MissingPromptHolderException,
        Logs retry attempts at the INFO level and stops after a maximum number of attempts.

        Args:
            current_seed: The template that is selected.

        Returns:
            converted template with placeholder for prompt.

        Raises:
            MissingPromptHolderException: If the prompt placeholder is empty after exhausting the maximum number of retries.
        """
        
        self._template_converter = random.choice(self._template_converter)

        target_seed_obj = await self._template_converter.convert_async(prompt=current_seed)
        if TEMPLATE_PLACEHOLDER not in target_seed_obj.output_text:
            raise MissingPromptHolderException(message="Prompt placeholder is empty.")
        return target_seed_obj

@dataclass
class FuzzerResult:
    jailbreak: bool
    nodes: PromptNode
    prompt_target_conversation_id: Optional[str] = None

    def __str__(self) -> str:
        return (
            "FuzzerResult("
            f"success={self.success}, "
            f"nodes={self.nodes}, "
            f"prompt_target_conversation_id={self.prompt_target_conversation_id})"
        )

    def __repr__(self) -> str:
        return self.__str__()


