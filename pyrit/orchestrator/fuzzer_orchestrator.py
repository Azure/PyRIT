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
                 # response: str = None,
                 # results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 ):
        self._fuzzer: 'FuzzerOrchestrator' = fuzzer
        self._template: str = template
       # self._response: str = response
       # self._results: 'list[int]' = results
        self._visited_num = 0

        self._parent: 'PromptNode' = parent
        self._children: 'list[PromptNode]' = []
        self._level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def _index(self):
        return self._index

    @index.setter
    def _index(self, index: int):
        self._index = index
        if self._parent is not None:
            self._parent._children.append(self)

    # @property
    # def _num_jailbreak(self):
    #     return sum(self._results)

    # @property
    # def _num_reject(self):
    #     return len(self._results) - sum(self._results)

    # @property
    # def _num_query(self):
    #     return len(self._results)

class FuzzerOrchestrator(Orchestrator):
    _memory: MemoryInterface

    DEFAULT_TEMPLATE_CONVERTER = [ShortenConverter(), ExpandConverter()] 
    TEMPLATE_PLACEHOLDER = '[INSERT PROMPT HERE]'

    def __init__(
        self,
        *,
        attack_content: list[str],   #questions in GPTFuzzer
        prompt_target: PromptTarget,
        prompt_templates: list[str], # list of all the jailbreak prompts on which MCTS will be applied.
        prompt_converters: Optional[list[PromptConverter]] = None,
        template_converter: Optional[list[PromptConverter]] = None, # shorten/expand 
        scorer: Scorer,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
        ratio=0.5, alpha=0.1, beta=0.2,
        random_state = None,
        max_jailbreak = -1, #Question: Should we get the max_jailbreak, max_query from user or a constant value?
        max_query = -1,
    ) -> None:

        """ Paper: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.

            Link: https://arxiv.org/pdf/2309.10253

            Author: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

            GitHub: https://github.com/sherdencooper/GPTFuzz/tree/master"""

        """Creates an orchestrator that explores a variety of jailbreak options via fuzzing.
        
        Args:
        
            attack_content: The attack content will be the questions to the target (Similar to jailbreak_prompt). 
            
            prompt_target: The target to send the prompts to.
            
            prompt_templates: List of all the jailbreak templates which will act as the seed pool. 
            At each iteration, a seed will be selected using the MCTS-explore algorithm which will be sent to the 
            shorten/expand prompt converter.
            
                          The converted prompt along with the attack_content will be sent to the target. 
                          
            prompt_converters: The prompt_converters to use to convert the prompts before sending 
            them to the prompt target. 
            
            template_converter: The converter that will be applied on the jailbreak template that was 
            selected by MCTS-explore. 
            
                          The prompt converters will not be applied to the attack_content. 
                          Shorten/expand prompt converters.
                          
            scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient (False)
                to satisfy the objective that is specified in the attack_strategy.
                
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory 
            will be used.
            
            memory_labels: The labels to use for the memory. This is useful to identify the messages in the memory.
            
            verbose: Whether to print debug information.

            ratio: constant that balances between the seed with high reward and the seed that is selected fewer times.

            alpha: Reward penalty. Reward penalty diminishes the reward for the current node and its ancestors when the path lengthens.

            beta: Minimal reward. Minimal reward prevents the reward of the current node and its ancestors from being too small or negative.

            max_jailbreak: maximum number of jailbreaks

            max_query: maximum number of query.

        """

        self._prompt_target = prompt_target
        self._achieved_objective = False
        self._attack_content = attack_content

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._memory = self._memory
        self._prompt_templates = prompt_templates
        self._template_converter = template_converter
        self._ratio = ratio  # balance between exploration and exploitation     #question: alpha, beta and ratio values are used for computing the rewards. vaile will not change. I don't think we need a input validation. 
        self._alpha = alpha  # penalty for level
        self._beta = beta   # minimal reward after penalty
        self._initial_prompts_nodes = self._prompt_nodes.copy()
        self._max_jailbreak = max_jailbreak
        self._max_query = max_query
        self._current_query = 0
        self._current_jailbreak = 0
        
        
        if not self._prompt_templates:
            raise ValueError("The initial seed cannot be empty.")
        
        if scorer.scorer_type != "true_false":
            raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {scorer.scorer_type}.")
        self._scorer = scorer

        self._prompt_nodes: 'list[PromptNode]' = [ #convert each template into a node and maintain the node information parent,child etc
            PromptNode(self, prompt) for prompt in prompt_templates 
        ]

        for i, prompt_node in enumerate(self._prompt_nodes):
            prompt_node.index = i

        if template_converter is None:
            template_converter = self.DEFAULT_TEMPLATE_CONVERTER
        else:
            template_converter = self._template_converter

        def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            # ('max_reject', 'current_reject'), #Question: should we need reject and iteration in the stopping criteria?
            # ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

        async def execute_fuzzer(
        self, *, prompt_templates: list[str] = None
        ) -> PromptRequestPiece:
            
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
            while not self.is_stop():
            
                # 1. Select a seed from the list of the templates using the MCTS
            
                current_seed = await self._get_seed(prompt_templates=prompt_templates)

                #2. Apply seed converter to the selected template.
            
                target_seed_obj = await self._template_converter.convert_async(prompt = current_seed)
                if TARGET_PLACEHOLDER not in target_seed_obj.output_text:
                    logger.info(f"Target placeholder is empty")
                    count = 0
                    while count < 4:
                        target_seed_obj = await self._template_converter.convert_async(prompt = current_seed)
                        if TARGET_PLACEHOLDER not in target_seed_obj.output_text:
                            logger.info(f"Target placeholder is empty")
                        else:
                            target_template = PromptTemplate(target_seed_obj.output_text)
                            break
                        count +=1
                    else:
                        raise ValueError("Invalid template.")
                    
                else:
                    target_template = PromptTemplate(target_seed_obj.output_text,parameters = list(**kwargs)) #check 

                target_template_node = PromptNode(self, target_template, parent= current_seed) # convert the target_template into a prompt_node to maintain the tree information
            

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

                responses = (   #list[PromptRequestResponse]: A list of PromptRequestResponse objects representing the responses received for each prompt.
                await self._prompt_normalizer.send_prompt_batch_to_target_async(
                    requests=requests,
                    target=self._prompt_target,
                    labels=self._global_memory_labels,
                    orchestrator_identifier=self.get_identifier(),
                    batch_size=self._batch_size, 
                    )
                )

                response_pieces = [response.request_pieces[0] for response in responses]
            
                #5. Apply scorer on the response and based on the scorer return if jailbreak successful or not.
                #categoryscorer
                #datasets/score/content_classifiers/content_classifier_system_prompt.yaml ?????? todo: can we use the same yaml file? refusal message ""I'm Sorry, I can't help you with that" can be expanded.

      
                scored_response = [await self._scorer.score_async(text=response)  for response in response_pieces]

                dic_val = {'reject' : 0, 'jailbreak': 1} 
                score_values = [dic_val[key] for key in dic_val]
 
                #6. Update the rewards for each of the node.
                self._num_jailbreak = sum(score_values)
                self._num_query = len(score_values)

                self._current_jailbreak += self._num_jailbreak
                self._current_query += self._num_query

                if self._num_jailbreak > 0: #successful jailbreak
                    self._prompt_nodes.append(target_template_node) # append the template to the initial template list.
                
            
                self._update(target_template_node) #update the rewards for the target_node


        def _get_seed(self, prompt_templates) -> PromptNode:
            self._step = 0 # to keep track of the steps or the count
            self._mctc_select_path: 'list[PromptNode]' = [] # type: ignore # keeps track of the path that has been currently selected
            self._last_choice_index = None
            self._rewards = []
            

            seed = _select(self)
            return seed

        def _select(self) -> PromptNode:
            self._step += 1
            if len(self._prompt_nodes) > len(self._rewards): # if the length of list of templates greater than rewards list (when new nodes added because of successful jailbreak) assign 0 as initial reward. 
                self._rewards.extend(
                    [0 for _ in range(len(self._prompt_nodes) - len(self._rewards))])

            current = max(  #initial path
                self._initial_prompts_nodes,
                key= self._best_UCT_score
            )
            self.mctc_select_path.append(current)

            while len(current._children) > 0: # while node is not a leaf 
                if np.random.rand() < self.alpha:
                    break
                current = max(   #compute the bestUCT score
                    current._children,
                    key= self._best_UCT_score
                )
                self._mctc_select_path.append(current) # append node to path

            for pn in self._mctc_select_path:
                pn._visited_num += 1    # keep track of number of visited nodes

            self._last_choice_index = current.index
            return current # returns the best child

    def _best_UCT_score(self):
        
        """Function to compute the score for each seed. The highest-scoring seed will be selected as the next seed.  
        
         This is an extension of the Monte-carlo tree search algorithm which applies Upper Confidence Bound (UCB) to the node. 
         
         UCB function determines the confidence interval for each node and returns the highest value which will be selected as next seed."""
        
        return self._rewards[_index] / (_visited_num + 1) +  #seed's average reward
                    self._ratio * np.sqrt(2 * np.log(self._step) / # self._ratio - constant that balances between the seed with high reward and the seed that is selected fewer times. 
                                     (_visited_num + 0.01))
                

    def _update(self, prompt_nodes: 'list[PromptNode]'): #have to fix this function based on the output of the template converter. 
        success_number = sum([prompt_node._num_jailbreak
                        for prompt_node in _prompt_nodes])    #Question: if output of template_converter is a str, the success_number will be always 1?

        last_choice_node = self._prompt_nodes[self._last_choice_index]
        for prompt_node in reversed(self._mctc_select_path):
            reward = success_number / (len(self._attack_content)
                                 * len(prompt_nodes))
            self._rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node._level))
            

#Question: in the GPTFuzzer paper output of a template_converter is a list of str. I assume that the current selected template(using MCTS algorithm) is given to template converter(shorten/expand) and the output will be a 
#template (str) that is shorten/expanded. I am doing the update for single template whereas in fuzzer paper for list of str. check the logic. 






    



        
        





