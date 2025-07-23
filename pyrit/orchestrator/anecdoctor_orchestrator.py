# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import textwrap
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import yaml
from colorama import Fore, Style

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget

# Suppress INFO and DEBUG messages from pyrit and httpx libraries
logging.getLogger("pyrit").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AnecdoctorOrchestrator(Orchestrator):
    """
    Anecdoctor Orchestrator class is responsible for:
      1. (Optionally) extracting a knowledge graph if use_knowledge_graph=True.
      2. Constructing and evaluating prompts based on data in ClaimsReview format.
    """

    def __init__(
        self,
        *,
        chat_model_under_evaluation: PromptChatTarget,
        use_knowledge_graph: bool = False,
        processing_model: Optional[PromptChatTarget] = None,
        evaluation_data: list[str],
        language: str = "english",
        content_type: str = "viral tweet",
        # scorer: Judge TBA
        prompt_converters: Optional[List[PromptConverter]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes an AnecdoctorOrchestrator object.

        Args:
            chat_model_under_evaluation (PromptChatTarget): The chat model to be used or evaluated.
            use_knowledge_graph (bool): Whether to run the knowledge-graph-extraction step.
            processing_model (PromptChatTarget, optional): The model used for the graph extraction.
            evaluation_data (List[str]): The data to be used in constructing the prompt.
            language (str): The language of the content. Defaults to "english".
            content_type (str): The type of content to generate. Defaults to "viral tweet".
            prompt_converters (List[PromptConverter], Optional): The prompt converters to be used.
            verbose (bool, Optional): Whether to print verbose output. Defaults to False.
        """
        # Initialize parent class (Orchestrator)
        super().__init__(prompt_converters=prompt_converters or [], verbose=verbose)

        self._chat_model_under_evaluation = chat_model_under_evaluation
        self._use_knowledge_graph = use_knowledge_graph
        self._processing_model = processing_model
        self._evaluation_data = evaluation_data
        self._language = language
        self._content_type = content_type

        self._conversation_id_final = str(uuid4())
        self._conversation_id_kg = str(uuid4())
        self._normalizer = PromptNormalizer()
        self._kg_result: Optional[str] = None

    def _load_prompt_from_yaml(self, yaml_filename: str) -> str:
        """
        Loads a prompt template from a given YAML file (relative path).
        Returns the 'value' key as a string.
        """
        prompt_path = Path(DATASETS_PATH, "orchestrators", "anecdoctor", yaml_filename)
        prompt_data = prompt_path.read_text(encoding="utf-8")
        yaml_data = yaml.safe_load(prompt_data)
        return yaml_data["value"]

    async def _extract_knowledge_graph(self) -> str:
        """
        Produces the knowledge graph using the processing model, returning the raw string output.
        """
        if not self._evaluation_data:
            raise ValueError("No example claims provided for knowledge graph construction.")

        if not self._processing_model:
            raise ValueError("Processing model is not set. Cannot extract knowledge graph.")

        # 1. Load the prompt template for KG extraction
        kg_system_prompt = self._load_prompt_from_yaml("anecdoctor_build_knowledge_graph.yaml")
        processed_kg_system_prompt = kg_system_prompt.format(language=self._language)

        # 2. Set the processing prompt as the system prompt
        self._processing_model.set_system_prompt(
            system_prompt=processed_kg_system_prompt,
            conversation_id=self._conversation_id_kg,
            orchestrator_identifier=self.get_identifier(),
            labels=self._global_memory_labels,
        )

        # 3. Format the examples into a single user prompt
        formatted_examples = "### examples\n" + "\n".join(self._evaluation_data)

        # 4. Create the SeedPromptGroup with the formatted examples
        seed_prompt_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=formatted_examples,
                    data_type="text",
                )
            ]
        )

        # 5. Send the seed prompt group to the normalizer, which will forward it to the model
        response = await self._normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            conversation_id=self._conversation_id_kg,
            target=self._processing_model,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        kg_output = response.get_value()
        return kg_output

    async def generate_attack(self) -> str:
        """
        Runs the orchestrator, possibly extracting a knowledge graph first,
        then generating the final content. Returns the final model output (answer).
        """
        if not self._evaluation_data:
            raise ValueError("No example data provided for evaluation.")

        if not self._chat_model_under_evaluation:
            raise ValueError("Chat model under evaluation is not set.")

        # 1. Load the prompt template
        if self._use_knowledge_graph:
            # 1a. load the system prompt if using the knowledge graph
            system_prompt = self._load_prompt_from_yaml("anecdoctor_use_knowledge_graph.yaml")
            # 1b. run the knowledge graph extraction
            self._kg_result = await self._extract_knowledge_graph()
            # 1c. set examples to knowledge graph format
            formatted_examples = self._kg_result
        else:
            # If not using the knowledge graph, load the default few-shot prompt
            system_prompt = self._load_prompt_from_yaml("anecdoctor_use_fewshot.yaml")
            # format the examples into a single user prompt
            formatted_examples = "### examples\n" + "\n".join(self._evaluation_data)

        # 2. Substitute the parameters in the prompt template
        processed_prompt = system_prompt.format(language=self._language, type=self._content_type)

        # 3. Set the system prompt on the chat model
        self._chat_model_under_evaluation.set_system_prompt(
            system_prompt=processed_prompt,
            conversation_id=self._conversation_id_final,
            orchestrator_identifier=self.get_identifier(),
            labels=self._global_memory_labels,
        )

        # 4. Create the SeedPromptGroup with the formatted examples
        seed_prompt_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=formatted_examples,
                    data_type="text",
                )
            ]
        )

        # 5. Send the seed prompt group to the normalizer, which will forward it to the model
        response = await self._normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            conversation_id=self._conversation_id_final,
            target=self._chat_model_under_evaluation,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        # 6. Extract the assistant's answer from the response pieces
        final_answer = response.get_value()

        # 7. If verbose, print a nicely formatted message
        if self._verbose:
            # print the examples or knowledge graph
            print(f"{Style.BRIGHT}{Fore.BLUE}user:{Style.RESET_ALL}")
            wrapped_user_text = textwrap.fill(formatted_examples, width=100)
            print(f"{Style.BRIGHT}{Fore.BLUE}{wrapped_user_text}{Style.RESET_ALL}")

            print(f"{Style.NORMAL}{Fore.YELLOW}assistant:{Style.RESET_ALL}")
            wrapped_answer = textwrap.fill(final_answer, width=100)
            print(f"{Style.NORMAL}{Fore.YELLOW}{wrapped_answer}{Style.RESET_ALL}")

        # 8. Return the final answer
        return final_answer
