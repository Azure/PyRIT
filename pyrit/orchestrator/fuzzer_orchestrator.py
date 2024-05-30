# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
from typing import Optional, Union, Dict, Any
from uuid import uuid4
from PIL import Image
from colorama import Fore, Style

from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.memory import MemoryInterface
from pyrit.models import AttackStrategy, PromptRequestPiece
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import NormalizerRequestPiece, PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score

logger = logging.getLogger(__name__)

class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete

#Class to maintain the tree information.
class PromptNode:
    def __init__(self,
                 fuzzer: 'FuzzerOrchestrator',
                 template: str,                #jailbreak template /prompt in GPTFuzzer
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 ):
        self.fuzzer: 'FuzzerOrchestrator' = fuzzer
        self.template: str = template
        self.response: str = response
        self.results: 'list[int]' = results
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)

class FuzzerOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_content: list[str],   #questions in GPTFuzzer
        prompt_target: PromptTarget,
        initial_seed: list[str], # list of all the jailbreak prompts on which MCTS will be applied.
        prompt_converters: Optional[list[PromptConverter]] = None,
        seed_converter: list[PromptConverter], # shorten/expand 
        scorer: Scorer,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
        ratio=0.5, alpha=0.1, beta=0.2,
    ) -> None:

        """Creates an orchestrator that explores a variety of jailbreak options via fuzzing.
        
        Args:
        
            attack_content: The attack content will be the questions to the target (Similar to jailbreak_prompt). 
            
            prompt_target: The target to send the prompts to.
            
            initial_seed: list of all the jailbreak templates which will act as the seed pool. 
            At each iteration, a seed will be selected using the MCTS-explore algorithm which will be sent to the 
            shorten/expand prompt converter.
            
                          The converted prompt along with the attack_content will be sent to the target. 
                          
            prompt_converters: The prompt_converters to use to convert the prompts before sending 
            them to the prompt target. 
            
            seed_converter: The converter that will be applied on the jailbreak template that was 
            selected by MCTS-explore. 
            
                          The prompt converters will not be applied to the attack_content. 
                          Shorten/expand prompt converters.
                          
            scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient (False)
                to satisfy the objective that is specified in the attack_strategy.
                
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory 
            will be used.
            
            memory_labels: The labels to use for the memory. This is useful to identify the messages in the memory.
            
            verbose: Whether to print debug information.

            Paper: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.

            Link: https://arxiv.org/pdf/2309.10253

            Author: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

            GitHub: https://github.com/sherdencooper/GPTFuzz/tree/master
        """

        self._prompt_target = prompt_target
        self._achieved_objective = False

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._memory = self._memory
        self._initial_seed = initial_seed
        self._seed_converter = seed_converter
        self._ratio = ratio  # balance between exploration and exploitation
        self._alpha = alpha  # penalty for level
        self._beta = beta   # minimal reward after penalty

        self._prompt_nodes: 'list[PromptNode]' = [ #convert each template into a node and maintain the node information parent,child etc
            PromptNode(self, prompt) for prompt in initial_seed 
        ]

        self._initial_prompts_nodes = self._prompt_nodes.copy()
        for i, prompt_node in enumerate(self._prompt_nodes):
            prompt_node.index = i

        if not self._initial_seed:
            raise ValueError("The initial seed cannot be empty.")
        if not self._seed_converters:
            raise ValueError("The prompt converter cannot be empty")
        
        if scorer.scorer_type != "true_false":
            raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {scorer.scorer_type}.")
        self._scorer = scorer

        async def send_prompt_async(
        self, *, initial_seed: list[str] = None, blocked: bool = False
        ) -> PromptRequestPiece:
            """Send the initial seed to the MCTS-explore to select a seed at each iteration

            Args:
                initial seed: A list of the initial jailbreak templates that needs to be sent to the MCTS algorithm. 

            """
            # target_messages = self._memory.get_chat_messages_with_conversation_id(
            # conversation_id=self._prompt_target_conversation_id
            # )  #message id?????

            #Steps:
            #1. select a seed
            #2. apply prompt converter
            #3. append prompt(questions in GPTFuzzer) with the selected template(seed). For each selected template append all the questions
            #4. send it to target and get a response
            #5. if jailbreak is successful then retain the template in seed pool. 
            #6. Update the rewards for each of the node. update()

            # 1. Select a seed from the list of the templates using the MCTS
            current_seed = await self._get_seed_usingMCTS(initial_seed=initial_seed)

            #2. apply prompt converter to the selected template. Apply seed converter.
            target_seed_obj = NormalizerRequestPiece(
            prompt_converters=self._seed_converter,
            prompt_text=current_seed,
            prompt_data_type="text",
            )

            #3. append prompt converted template with prompt. Apply each of the prompts (questions) to the template. 
            #Todo: should I write a separate orchestrator to do this (similar to PromptSendingOrchestrator)?
            

            #4. Apply prompt converter if any and Send request to a target 
              target_prompt_obj = NormalizerRequestPiece(
            prompt_converters=self._prompt_converters,
            prompt_text=prompt,
            prompt_data_type="text",
            )
            
            response_piece = (
            await self._prompt_normalizer.send_prompt_async(
                normalizer_request=NormalizerRequest([target_prompt_obj]),
                target=self._prompt_target,
                conversation_id=self._prompt_target_conversation_id,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
            )
        ).request_pieces[0]
            
            #5. Todo: Apply scorer on the response and based on the scorer return if jailbreak successful or not.

            #6. Update the rewards for each of the node. 
            update(PromptNode) #fix this




        def _get_seed_usingMCTS(self, initial_seed) -> PromptNode:
            self._step = 0 # to keep track of the steps or the count
            self._mctc_select_path: 'list[PromptNode]' = [] # type: ignore # keeps track of the path that has been currently selected
            self._last_choice_index = None
            self._rewards = []
            

            seed = select(self)
            return seed

        def select(self) -> PromptNode:
            self._step += 1
            if len(self.fuzzer._prompt_nodes) > len(self._rewards):
                self._rewards.extend(
                    [0 for _ in range(len(self.fuzzer._prompt_nodes) - len(self._rewards))])

        self._mctc_select_path.clear()
        cur = max(  #initial path
            self.fuzzer._initial_prompts_nodes,
            key=lambda pn:
            self._rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0: # while node is not a leaf 
            if np.random.rand() < self.alpha:
                break
            cur = max(   #compute the bestUCT score
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self._ratio * np.sqrt(2 * np.log(self._step) /
                                     (pn.visited_num + 0.01))
            )
            self._mctc_select_path.append(cur) # append node to path

        for pn in self._mctc_select_path:
            pn.visited_num += 1    # keep track of number of visited nodes

        self._last_choice_index = cur.index
        return cur # returns the best child

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in _prompt_nodes])

        last_choice_node = self.fuzzer._prompt_nodes[self._last_choice_index]
        for prompt_node in reversed(self._mctc_select_path):
            reward = succ_num / (len(self.fuzzer.questions)
                                 * len(prompt_nodes))
            self._rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))
            








    



        
        





