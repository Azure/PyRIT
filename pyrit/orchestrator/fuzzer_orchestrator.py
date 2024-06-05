# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
from typing import Optional, Union, Dict, Any
from uuid import uuid4
from colorama import Fore, Style

from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.memory import MemoryInterface
from pyrit.models import AttackStrategy, PromptRequestPiece
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import NormalizerRequestPiece, PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestionPaths

logger = logging.getLogger(__name__)

class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete

#Class to maintain the tree information.
#todo remove variables and exports that are not necessary
class PromptNode:
    def __init__(self,
                 fuzzer: 'FuzzerOrchestrator',
                 template: str,                #jailbreak template /prompt in GPTFuzzer
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 ):
        self._fuzzer: 'FuzzerOrchestrator' = fuzzer
        self._template: str = template
        self._response: str = response
        self._results: 'list[int]' = results
        self._visited_num = 0

        self._parent: 'PromptNode' = parent
        self._child: 'list[PromptNode]' = []
        self._level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def _index(self):
        return self._index

    @index.setter
    def _index(self, index: int):
        self._index = index
        if self._parent is not None:
            self._parent._child.append(self)

    @property
    def _num_jailbreak(self):
        return sum(self._results)

    @property
    def _num_reject(self):
        return len(self._results) - sum(self._results)

    @property
    def _num_query(self):
        return len(self._results)

class FuzzerOrchestrator(Orchestrator):
    _memory: MemoryInterface

    DEFAULT_SEED_CONVERTER = "shorten/expand" # fix this which one of this should be the default converter if the user given option is None?

    def __init__(
        self,
        *,
        attack_content: list[str],   #questions in GPTFuzzer
        prompt_target: PromptTarget,
        initial_seed: list[str], # list of all the jailbreak prompts on which MCTS will be applied.
        prompt_converters: Optional[list[PromptConverter]] = None,
        seed_converter: Optional[list[PromptConverter]] = None, # shorten/expand 
        scorer: Scorer,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
        ratio=0.5, alpha=0.1, beta=0.2,
    ) -> None:

        """ Paper: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.

            Link: https://arxiv.org/pdf/2309.10253

            Author: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

            GitHub: https://github.com/sherdencooper/GPTFuzz/tree/master"""

        """Creates an orchestrator that explores a variety of jailbreak options via fuzzing.
        
        Args:
        
            attack_content: The attack content will be the questions to the target (Similar to jailbreak_prompt). 
            
            prompt_target: The target to send the prompts to.
            
            initial_seed: List of all the jailbreak templates which will act as the seed pool. 
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
        self._initial_prompts_nodes = self._prompt_nodes.copy()
        
        
        if not self._initial_seed:
            raise ValueError("The initial seed cannot be empty.")
        
        if scorer.scorer_type != "true_false":
            raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {scorer.scorer_type}.")
        self._scorer = scorer

        self._prompt_nodes: 'list[PromptNode]' = [ #convert each template into a node and maintain the node information parent,child etc
            PromptNode(self, prompt) for prompt in initial_seed 
        ]

        for i, prompt_node in enumerate(self._prompt_nodes):
            prompt_node.index = i

       #set the default seed_converter. Seed_converter= Optional[list[PromptConverter]] the list can have stacked converter. But stacked converters are not allowed. Fix this.
        if seed_converter is None:
            seed_converter = self.DEFAULT_SEED_CONVERTER
        else:
            seed_converter = self._seed_converter

        async def execute_fuzzer(
        self, *, initial_seed: list[str] = None
        ) -> PromptRequestPiece:
            
            """#Steps:
            #1. Select a seed - Send the initial seed to the MCTS-explore to select a seed at each iteration
            
            #2. Apply seed converter
            
            #3. append prompt(questions in GPTFuzzer) with the selected template(seed). For each selected template append all the questions
            
            #4. Apply prompt converters if any and send it to target and get a response
            
            #5. If jailbreak is successful then retain the template in seed pool.
            
            #6. Update the rewards for each of the node. update()

            Args:
                initial seed: A list of the initial jailbreak templates that needs to be sent to the MCTS algorithm. 

            """
            # todo: define the stopping criteria. In the fuzzer paper, their stopping criteria is based on max_query(defaults to 1000, user can also specify), max_jailbreak(defaults to 1, user can specify), 
            #Evaluated using single-question and multi-question attack. What should be our stopping criteria?
            
            # 1. Select a seed from the list of the templates using the MCTS
            
            current_seed = await self._get_seed(initial_seed=initial_seed)

            #2. Apply seed converter to the selected template.
            
            target_seed_obj = await self._seed_converter.convert_async(prompt = current_seed)
            target_template = PromptTemplate(target_seed_obj.output_text)

            #3. Append prompt converted template with prompt. Apply each of the prompts (questions) to the template. 
            
             jailbreak_prompts = []
             for prompt in attack_content:
                jailbreak_prompts.append(
                    target_template.apply_custom_metaprompt_parameters(prompt=prompt) 
                )

            #4. Apply prompt converter if any and send request to a target 
            
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

            response_piece = (   #list[PromptRequestResponse]: A list of PromptRequestResponse objects representing the responses received for each prompt.
            await self._prompt_normalizer.send_prompt_batch_to_target_async(
                requests=requests,
                target=self._prompt_target,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
                batch_size=self._batch_size, 
                )
            )
            
            #5. Apply scorer on the response and based on the scorer return if jailbreak successful or not.
            #true/false or categoryscorer
            #datasets/score/content_classifiers/content_classifier_system_prompt.yaml ?????? todo: can we use the same yaml file? refusal message ""I'm Sorry, I can't help you with that" can be expanded.

            content_classifier = SelfAskCategoryScorer( # change the path
                             content_classifier=ContentClassifierPaths.CONTENT_CLASSIFIER_SYSTEM_PROMPT.value, chat_target=self._prompt_target
                             )
            scored_response = await content_classifier.score_async(text=response_piece) 
            
            # true_false_classifier = SelfAskTrueFalseScorer( # change the path
            #                 true_false_question_path=TrueFalseQuestionPaths.PROMPT_INJECTION.value, chat_target=self._prompt_target
            #                 )
            # scored_response = await true_false_classifier.score_async(text=response_piece) 

            #6. Update the rewards for each of the node.
            
            _update(PromptNode) #fix this # todo: have to update the rewards by calling _update(). Also update the nodes based on the successful jailbreaks.


        def _get_seed(self, initial_seed) -> PromptNode:
            self._step = 0 # to keep track of the steps or the count
            self._mctc_select_path: 'list[PromptNode]' = [] # type: ignore # keeps track of the path that has been currently selected
            self._last_choice_index = None
            self._rewards = []
            

            seed = _select(self)
            return seed

        def _select(self) -> PromptNode:
            self._step += 1
            if len(self._prompt_nodes) > len(self._rewards):
                self._rewards.extend(
                    [0 for _ in range(len(self._prompt_nodes) - len(self._rewards))])

            self._mctc_select_path.clear()
            current = max(  #initial path
                self._initial_prompts_nodes,
                key= _bestUCT_score(self)
            )
            self.mctc_select_path.append(current)

            while len(current._child) > 0: # while node is not a leaf 
                if np.random.rand() < self.alpha:
                    break
                current = max(   #compute the bestUCT score
                    current._child,
                    key= _bestUCT_score(self)
                )
                self._mctc_select_path.append(current) # append node to path

            for pn in self._mctc_select_path:
                pn._visited_num += 1    # keep track of number of visited nodes

            self._last_choice_index = current.index
            return current # returns the best child

    def _bestUCT_score(self):
        
        """Function to compute the score for each seed. The highest-scoring seed will be selected as the next seed.  
        
         This computation is based on UCB (Upper Confidence Bound) algorithm. It uses uncertainity in the estimates to drive exploration."""
        
        return lambda pn:
                    self._rewards[pn._index] / (pn._visited_num + 1) +  #seed's average reward
                    self._ratio * np.sqrt(2 * np.log(self._step) / # self._ratio - constant that balances between the seed with high reward and the seed that is selected fewer times. 
                                     (pn._visited_num + 0.01))
                

    def _update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node._num_jailbreak
                        for prompt_node in _prompt_nodes])

        last_choice_node = self._prompt_nodes[self._last_choice_index]
        for prompt_node in reversed(self._mctc_select_path):
            reward = succ_num / (len(self._fuzzer.questions) #fix this, wrong. 
                                 * len(prompt_nodes))
            self._rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node._level))
            








    



        
        





