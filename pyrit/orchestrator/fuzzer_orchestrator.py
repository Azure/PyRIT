# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import random
import numpy as np
from typing import Optional
from uuid import uuid4
from colorama import Fore, Style

from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score
from pyrit.score import SelfAskCategoryScorer
from pyrit.score.self_ask_category_scorer import ContentClassifierPaths
from pyrit.models import PromptTemplate
from typing import Callable
from pyrit.prompt_converter import shorten_converter,expand_converter

from pyrit.exceptions import EmptyResponseException, pyrit_retry

logger = logging.getLogger(__name__)

class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete

#Class to maintain the tree information.
#todo remove variables and exports that are not necessary
class PromptNode:
    def __init__(self,
                 #fuzzer: 'FuzzerOrchestrator',
                 template: str,                #jailbreak template /prompt in GPTFuzzer
                 # response: str = None,
                 # results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 ):
        #self._fuzzer: 'FuzzerOrchestrator' = fuzzer
        self._template: str = template
        
       # self._response: str = response
       # self._results: 'list[int]' = results
       
        

        self._parent: 'PromptNode' = parent
        self._children: 'list[PromptNode]' = []
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

    #DEFAULT_TEMPLATE_CONVERTER = [shorten_converter.ShortenConverter(), expand_converter.ExpandConverter()] 
    

    def __init__(
        self,
        *,
        prompts: list[str],   #questions in GPTFuzzer
        prompt_target: PromptTarget,
        prompt_templates: list[str], # list of all the jailbreak prompts on which MCTS will be applied.
        prompt_converters: Optional[list[PromptConverter]] = None,
        template_converter:list[PromptConverter], # shorten/expand 
        scorer: Scorer,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
        frequency_weight=0.5, reward_penalty=0.1, minimum_reward=0.2,
        non_leaf_nodeprobability =0.1,
        random_state = None,
        batch_size: int = 10,
    ) -> None:

        """Creates an orchestrator that explores a variety of jailbreak options via fuzzing.
        
        Paper: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.

            Link: https://arxiv.org/pdf/2309.10253

            Author: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

            GitHub: https://github.com/sherdencooper/GPTFuzz/tree/master
        
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
                          
            scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient (False)
                to satisfy the objective that is specified in the prompt.
                
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory 
            will be used.
            
            memory_labels: The labels to use for the memory. This is useful to identify the messages in the memory.
            
            verbose: Whether to print debug information.

            frequency_weight: constant that balances between the seed with high reward and the seed that is selected fewer times.

            Reward penalty: Reward penalty diminishes the reward for the current node and its ancestors when the path lengthens.

            Minimum reward. Minimal reward prevents the reward of the current node and its ancestors from being too small or negative.

            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.

        """

        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target
        self._achieved_objective = False
        self._prompts = prompts

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._memory = self._memory
        self._prompt_templates = prompt_templates
        self._template_converter = template_converter
        self._frequency_weight = frequency_weight  # balance between exploration and exploitation     
        self._reward_penalty = reward_penalty  # penalty for level
        self._minimum_reward = minimum_reward   # minimal reward after penalty
        self._non_leaf_nodeprobability = non_leaf_nodeprobability
        self._max_jailbreak = 1
        #self._max_query = len(self._prompt_templates)  * 10 #* len(self._prompts.prompts)
        self._max_query = 100
        self._current_query = 0
        self._current_jailbreak = 0
        self._batch_size = batch_size
        
        
        if not self._prompt_templates:
            raise ValueError("The initial seed cannot be empty.")
        
        if scorer.scorer_type != "true_false":
            raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {scorer.scorer_type}.")
        self._scorer = scorer

        self._prompt_nodes: 'list[PromptNode]' = [ #convert each template into a node and maintain the node information parent,child etc
            PromptNode(prompt) for prompt in prompt_templates 
        ]

        self._initial_prompts_nodes = self._prompt_nodes.copy()

        for i, prompt_node in enumerate(self._prompt_nodes):
            prompt_node.index = i
        
        self._template_converter = random.choice(self._template_converter)

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
        # while not self.is_stop():
        if self._current_query >= self._max_query:
            logger.info(f"Maximum query limit reached.")
        elif self._current_jailbreak >= self._max_jailbreak:
            logger.info(f"Maximum number of jailbreaks reached.")
        else:
            # 1. Select a seed from the list of the templates using the MCTS
            
            current_seed = await self._get_seed(prompt_templates=prompt_templates)

            #2. Apply seed converter to the selected template.
            try: 
                target_seed_obj = await self._apply_template_converter(current_seed._template)
            except:
                raise ValueError("Invalid template")
                
            target_template = PromptTemplate(target_seed_obj.output_text,parameters = ["prompt"])

            target_template_node = PromptNode(template = target_seed_obj, parent= current_seed) # convert the target_template into a prompt_node to maintain the tree information

            #3. Append prompt converted template with prompt. Apply each of the prompts (questions) to the template. 
            
            jailbreak_prompts = []
            for prompt in self._prompts:
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

            results = []

            scored_response = [] #check
            for response in response_pieces:
                scored_response.append(
                    self._scorer.score_async(response))
            
            batch_scored_response = await asyncio.gather(*scored_response)

            score_values = [score[0].get_value() for score in batch_scored_response]
 
            #6. Update the rewards for each of the node.
           # self._num_jailbreak = sum(score_values)
            self._num_jailbreak = score_values.count(True)
            self._num_query = len(score_values)

            self._current_jailbreak += self._num_jailbreak
            self._current_query += self._num_query

            if self._num_jailbreak > 0: #successful jailbreak
                self._prompt_nodes.append(target_template_node) # append the template to the initial template list.
                
            
            self._update(target_template_node) #update the rewards for the target_node


    async def _get_seed(self, prompt_templates) -> PromptNode:
        self._step = 0 # to keep track of the steps or the count
        self._last_choice_index = None
        self._mctc_select_path: 'list[PromptNode]' = [] # type: ignore # keeps track of the path that has been currently selected
        self._rewards = []
            

        seed = self._select()
        return seed

    def _select(self) -> PromptNode:
        self._step += 1
        if len(self._prompt_nodes) > len(self._rewards): # if the length of list of templates greater than rewards list (when new nodes added because of successful jailbreak) assign 0 as initial reward. 
            self._rewards.extend(
                [0 for _ in range(len(self._prompt_nodes) - len(self._rewards))])

        current = max(  #initial path
            self._initial_prompts_nodes,
            key = self._best_UCT_score()
        )
        self._mctc_select_path.append(current)

        while len(current._children) > 0: # while node is not a leaf 
            if np.random.rand() < self._non_leaf_nodeprobability:
                break
            current = max(   #compute the bestUCT score
                current._children,
                key= self._best_UCT_score()
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
        
        return lambda pn: self._rewards[pn.index] / (pn._visited_num + 1) +  \
            self._frequency_weight * np.sqrt(2 * np.log(self._step) /  (pn._visited_num + 0.01)) # self._frequency_weight - constant that balances between the seed with high reward and the seed that is selected fewer times. 
       

    def _update(self, prompt_nodes: 'PromptNode'): 
        """
        Function to update the reward of all the nodes in the last chosen path. 
        """
        success_number = self._num_jailbreak #check 

        last_chosen_node = self._prompt_nodes[self._last_choice_index]
        for prompt_node in reversed(self._mctc_select_path):
            reward = success_number / (len(self._prompts)
                                 * 1) # len(prompt_nodes))
            self._rewards[prompt_node.index] += reward * \
                max(self._minimum_reward, (1 - self._reward_penalty * last_chosen_node._level))
            
    @pyrit_retry 
    async def _apply_template_converter(self,current_seed):
        """
        Asynchronously applies template converter.

        Retries the function if it raises EmptyResponseException,
        Logs retry attempts at the INFO level and stops after a maximum number of attempts.

        Args:
            current_seed: The template that is selected. 

        Returns:
            converted template with placeholder for prompt. 

        Raises:
            EmptyResponseException: If the prompt placeholder is empty after exhausting the maximum number of retries.
        """
        TEMPLATE_PLACEHOLDER = '{{ prompt }}' #check

        target_seed_obj = await self._template_converter.convert_async(prompt = current_seed)
        if TEMPLATE_PLACEHOLDER not in target_seed_obj.output_text:
            raise EmptyResponseException(message="Prompt placeholder is empty.")
        return target_seed_obj


   






    



        
        





