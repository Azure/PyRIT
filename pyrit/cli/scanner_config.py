# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
from copy import deepcopy
from importlib import import_module
from typing import Any, List, Literal, Optional, Type, get_args

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from pyrit.common.initialization import MemoryDatabaseType
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration

SupportedExecutionTypes = Literal["local"]


def load_class(module_name: str, class_name: str, error_context: str) -> Type[Any]:
    """
    Dynamically import a class from a module by name.
    """
    try:
        mod = import_module(module_name)
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            raise TypeError(f"The attribute {class_name} in module {module_name} is not a class.")
    except Exception as ex:
        raise RuntimeError(f"Failed to import {class_name} from {module_name} for {error_context}: {ex}") from ex

    return cls


class DatabaseConfig(BaseModel):
    """
    Configuration for the database used by the scanner.
    """

    db_type: MemoryDatabaseType = Field(
        ...,
        alias="type",
        description=f"Which database to use. Supported values: {list(get_args(MemoryDatabaseType))}",
    )
    memory_labels: dict = Field(default_factory=dict, description="Labels that will be stored in memory to tag runs.")


class ScenarioConfig(BaseModel, extra="allow"):
    """
    Configuration for a single scenario orchestrator.
    """

    scenario_type: str = Field(
        ..., alias="type", description="Scenario orchestrator class/type (e.g. 'PromptSendingOrchestrator')."
    )

    @model_validator(mode="after")
    def check_scenario_type(self) -> "ScenarioConfig":
        """
        Robustness check to ensure the user actually provided a scenario_type in the YAML.
        Pydantic already enforces requiredness, but we are adding more checks here.
        """
        if not self.scenario_type:
            raise ValueError("Scenario 'type' must not be empty.")
        return self

    def create_orchestrator(
        self,
        objective_target: Any,
        adversarial_chat: Optional[Any] = None,
        prompt_converters: Optional[List[Any]] = None,
        scoring_target: Optional[Any] = None,
        objective_scorer: Optional[Any] = None,
    ) -> Any:
        """
        Load and instantiate the orchestrator class,
        injecting top-level objects (targets, scorers) as needed.
        """
        # Loading the orchestrator class by name, e.g 'RedTeamingOrchestrator'
        orchestrator_class = load_class(
            module_name="pyrit.orchestrator", class_name=self.scenario_type, error_context="scenario"
        )

        # Converting scenario fields into a dict for the orchestrator constructor
        scenario_args = self.model_dump(exclude={"scenario_type"})
        scenario_args = deepcopy(scenario_args)

        # Inspecting the orchestrator constructor so we can inject the optional arguments if they exist
        constructor_arg_names = [
            param.name for param in inspect.signature(orchestrator_class.__init__).parameters.values()
        ]

        # Building a map of complex top-level objects that belong outside the scenario
        complex_args = {
            "objective_target": objective_target,
            "adversarial_chat": adversarial_chat,
            "scoring_target": scoring_target,
            "objective_scorer": objective_scorer,
        }

        # Disallowing scenario-level overrides for these complex args
        for key in complex_args:
            if key in scenario_args:
                raise ValueError(f"{key} must be configured at the top-level of the config, not inside a scenario.")

        # If the orchestrator constructor expects any of these, inject them
        for key, value in complex_args.items():
            if key in constructor_arg_names and value is not None:
                scenario_args[key] = value

        # Handle converters: prefer request_converter_configurations if present, else prompt_converters
        if "request_converter_configurations" in constructor_arg_names:
            if prompt_converters:
                scenario_args["request_converter_configurations"] = PromptConverterConfiguration.from_converters(
                    converters=prompt_converters
                )
        elif "prompt_converters" in constructor_arg_names:
            scenario_args["prompt_converters"] = prompt_converters

        # And the instantiation of the orchestrator
        try:
            return orchestrator_class(**scenario_args)
        except Exception as ex:
            raise ValueError(f"Failed to instantiate scenario '{self.scenario_type}': {ex}") from ex


class TargetConfig(BaseModel):
    """
    Configuration for a prompt target (e.g. OpenAIChatTarget).
    """

    class_name: str = Field(..., alias="type", description="Prompt target class name (e.g. 'OpenAIChatTarget').")

    def create_instance(self) -> Any:
        """
        Dynamically instantiate the underlying target class.
        """
        target_class = load_class(
            module_name="pyrit.prompt_target", class_name=self.class_name, error_context="TargetConfig"
        )

        init_kwargs = self.model_dump(exclude={"class_name"})
        return target_class(**init_kwargs)


class ObjectiveScorerConfig(BaseModel):
    """
    Configuration for an objective scorer
    """

    type: str = Field(..., description="Scorer class (e.g. 'SelfAskRefusalScorer').")

    def create_scorer(self, scoring_target_obj: Optional[Any]) -> Any:
        """
        Load and instantiate the scorer class.
        """
        scorer_class = load_class(module_name="pyrit.score", class_name=self.type, error_context="objective_scorer")

        init_kwargs = self.model_dump(exclude={"type"})
        signature = inspect.signature(scorer_class.__init__)

        chat_target_key: str = "chat_target"
        if chat_target_key in signature.parameters:
            if scoring_target_obj is None:
                raise KeyError(
                    "Scorer requires a scoring_target to be defined. "
                    "Alternatively, the adversarial_target can be used for scoring purposes, "
                    "but none was provided."
                )
            init_kwargs[chat_target_key] = scoring_target_obj

        return scorer_class(**init_kwargs)


class ScoringConfig(BaseModel):
    """
    Configuration for the scoring setup, including optional
    override of the default adversarial chat with a 'scoring_target'
    and/or an 'objective_scorer'.
    """

    scoring_target: Optional[TargetConfig] = Field(
        None, description="If provided, use this target for scoring instead of 'adversarial_chat'."
    )
    objective_scorer: Optional[ObjectiveScorerConfig] = Field(
        None, description="Details for the objective scorer, if any."
    )

    def create_objective_scorer(self, scoring_target_obj: Optional[Any]) -> Optional[Any]:
        # If the user did not provide an objective_scorer config block (meaning the YAML lacks that section),
        # we simply return None – no scorer to instantiate.
        if not self.objective_scorer:
            return None

        return self.objective_scorer.create_scorer(scoring_target_obj=scoring_target_obj)


class ConverterConfig(BaseModel, extra="allow"):
    """
    Configuration for a single prompt converter, e.g. type: "Base64Converter"
    """

    class_name: str = Field(..., alias="type", description="The prompt converter class name (e.g. 'Base64Converter').")

    converter_target: Optional[TargetConfig] = Field(
        None, description="If provided, use this target for the converter LLM instead of 'adversarial_chat'."
    )

    def create_instance(self, converter_target: Optional[Any]) -> Any:
        """
        Dynamically load and instantiate the converter class
        """
        converter_class = load_class(
            module_name="pyrit.prompt_converter", class_name=self.class_name, error_context="prompt_converter"
        )

        init_kwargs = self.model_dump(exclude={"class_name", "converter_target"})
        signature = inspect.signature(converter_class.__init__)

        converter_target_key: str = "converter_target"
        if converter_target_key in signature.parameters:
            if converter_target is None:
                raise KeyError(
                    "Converter requires a converter_target to be defined. "
                    "Alternatively, the adversarial_target can be used for scoring purposes, "
                    "but none was provided."
                )
            init_kwargs[converter_target_key] = converter_target

        return converter_class(**init_kwargs)


class ExecutionSettings(BaseModel):
    """
    Configuration for how the scanner is executed (e.g. locally or via AzureML).
    """

    type: SupportedExecutionTypes = Field(
        "local", description=f"Execution environment. Supported values: {list(get_args(SupportedExecutionTypes))}"
    )
    parallel_nodes: Optional[int] = Field(None, description="How many scenarios to run in parallel.")


class ScannerConfig(BaseModel):
    """
    Top-level configuration for the entire scanner.
    """

    datasets: List[str] = Field(..., description="List of dataset YAML paths to load seed prompts from.")
    scenarios: List[ScenarioConfig] = Field(..., description="List of scenario orchestrators to execute.")
    objective_target: TargetConfig = Field(..., description="Configuration of the main (objective) chat target.")
    adversarial_chat: Optional[TargetConfig] = Field(
        None, description="Configuration of the adversarial chat target (if any)."
    )
    scoring: Optional[ScoringConfig] = Field(None, description="Scoring configuration (if any).")
    converters: Optional[List[ConverterConfig]] = Field(None, description="List of prompt converters to apply.")
    execution_settings: ExecutionSettings = Field(
        default_factory=lambda: ExecutionSettings.model_validate({}),
        description="Settings for how the scan is executed.",
    )
    database: DatabaseConfig = Field(
        ...,
        description="Database configuration for storing memory or results, including memory_labels.",
    )

    @field_validator("objective_target", mode="before")
    def check_objective_target_is_dict(cls, value):
        """
        Ensure the user actually provides a dict.
        Pydantic will run this validator before it attempts to parse the value into the TargetConfig model
        """
        if not isinstance(value, dict):
            raise ValueError(
                "Field 'objective_target' must be a dictionary.\n"
                "Example:\n"
                "  objective_target:\n"
                "    type: OpenAIChatTarget"
            )
        return value

    @model_validator(mode="after")
    def fill_scoring_target(self) -> "ScannerConfig":
        """
        If config.scoring exists but doesn't explicitly define a scoring_target,
        default it to the adversarial_chat
        """
        if self.scoring:
            if self.scoring.scoring_target is None and self.adversarial_chat is not None:
                self.scoring.scoring_target = self.adversarial_chat
        return self

    @model_validator(mode="after")
    def fill_converter_target(self) -> "ScannerConfig":
        """
        If config.converters are provided but don't explicitly define a converter_target,
        default it to the adversarial_chat
        """
        if self.converters:
            for converter_cfg in self.converters:
                # Check if converter takes converter target
                converter_class = load_class(
                    module_name="pyrit.prompt_converter",
                    class_name=converter_cfg.class_name,
                    error_context="prompt_converter",
                )

                signature = inspect.signature(converter_class.__init__)
                converter_target_key: str = "converter_target"

                # If the converter takes a converter target and it is not set, set it to the adversarial chat
                if (
                    converter_target_key in signature.parameters
                    and converter_cfg.converter_target is None
                    and self.adversarial_chat is not None
                ):
                    converter_cfg.converter_target = self.adversarial_chat

        return self

    @classmethod
    def from_yaml(cls, path: str) -> "ScannerConfig":
        """
        Loads configuration from a YAML file and validates it using Pydantic.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
        return cls(**raw_dict)

    def create_objective_scorer(self) -> Optional[Any]:
        """
        if there's an objective scorer configured,
        instantiate it using 'scoring_target' (which might be adversarial_chat).
        """
        if not self.scoring:
            return None

        scoring_target = None
        if self.scoring.scoring_target:
            scoring_target = self.scoring.scoring_target.create_instance()
        return self.scoring.create_objective_scorer(scoring_target_obj=scoring_target)

    def create_prompt_converters(self) -> List[PromptConverter]:
        """
        Instantiates each converter defined in 'self.converters' (if any).
        Returns a list of converter objects.
        """
        if not self.converters:
            return []
        instances = []

        converter_target = None
        for converter_cfg in self.converters:
            if converter_cfg.converter_target:
                converter_target = converter_cfg.converter_target.create_instance()
            instances.append(converter_cfg.create_instance(converter_target=converter_target))
        return instances

    def create_orchestrators(
        self, prompt_converters: Optional[list[PromptConverter] | list[PromptConverterConfiguration]] = None
    ) -> list[Any]:
        """
        Helper method to instantiate all orchestrators from the scenario configs,
        injecting objective_target, adversarial_chat, scoring_target, objective_scorer, etc.
        """
        # Instantiate the top-level targets
        objective_target_obj = self.objective_target.create_instance()
        adversarial_chat_obj = None
        if self.adversarial_chat:
            adversarial_chat_obj = self.adversarial_chat.create_instance()

        # If there is a scoring_target or an objective_scorer:
        scoring_target_obj = None
        objective_scorer_obj = None
        if self.scoring:
            # fill_scoring_target might have already assigned it to self.scoring.scoring_target
            if self.scoring.scoring_target:
                scoring_target_obj = self.scoring.scoring_target.create_instance()
            # create the actual scorer
            objective_scorer_obj = self.scoring.create_objective_scorer(scoring_target_obj=scoring_target_obj)

        # Now each scenario can create its orchestrator
        orchestrators = []
        for scenario in self.scenarios:
            orch = scenario.create_orchestrator(
                objective_target=objective_target_obj,
                adversarial_chat=adversarial_chat_obj,
                prompt_converters=prompt_converters,
                scoring_target=scoring_target_obj,
                objective_scorer=objective_scorer_obj,
            )
            orchestrators.append(orch)
        return orchestrators
