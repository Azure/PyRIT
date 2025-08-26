import abc
from pathlib import Path
from dataclasses import dataclass
import yaml
from pyrit.common.path import LIKERT_SCALES_PATH
from pyrit.models import SeedPrompt

@dataclass(frozen=True)
class ScorerConfig(abc.ABC):
    """
    Object that stores local variables extracted from YAML files; one per scorer.
    Exists to decouple scorer configuration and file handling from actual scoring logic.
    Responsibilities:
    1. Ensure all referenced YAML files exist.
    2. Provide all variables whose values are derived from YAML files to the scorer, with type
       and variable name guarantees (frozen).
    3. Validate configuration before proceeding to scoring (resolve contradictions, etc.)
    """
    def __init__(self) -> None:
        pass

    @property
    def vars(self) -> dict[str, str]:
        """
        Returns a dictionary of all variables whose values are derived from YAML files.
        """
        pass

    @abc.abstractmethod
    def validate(self) -> None:
        """
        Validates the configuration and reads the YAML files.
        This does not violate the dataclass contract, because all attributes are populated strictly
        after this completes.
        """
        pass

    @classmethod
    def path(cls, **kwargs) -> None:
        """
        Path parsing method for one or more YAML files, meant to be generic.
        """
        pass

class SelfAskLikertScorerConfig(ScorerConfig):


    def __init__(self, likert_scale_path: Path):
        super().__init__()
        self.validate(likert_scale_path)
    
    def validate(self, likert_scale_path: Path):
        if not likert_scale_path.exists():
            raise FileNotFoundError(f"Likert scale file not found: {likert_scale_path}")
        # extract category and scale descriptions
        likert_scale = yaml.safe_load(likert_scale_path.read_text(encoding="utf-8"))

        if likert_scale["category"]:
            self._score_category = likert_scale["category"]
        else:
            raise ValueError(f"Improperly formatted likert scale yaml file. Missing category in {likert_scale_path}.")

        descriptions = likert_scale["scale_descriptions"]
        if not descriptions:
            raise ValueError("Improperly formatted Likert scale yaml file. No likert scale_descriptions provided")

        likert_scale_description = ""

        for description in descriptions:
            name = description["score_value"]
            desc = description["description"]

            if int(name) < 0 or int(name) > 5:
                raise ValueError(
                    "Improperly formatted Likert scale yaml file. Likert scale values must be between 1 and 5"
                )

            likert_scale_description += f"'{name}': {desc}\n"

        self._scale_descriptions = likert_scale_description

        self._scoring_instructions_template = SeedPrompt.from_yaml_file(
            LIKERT_SCALES_PATH / "likert_system_prompt.yaml"
        )

        self._system_prompt = self._scoring_instructions_template.render_template_value(
            likert_scale=likert_scale, category=self._score_category
        )

    @property
    def system_prompt(self) -> str:
        if self._system_prompt not str:
            raise ValueError("System prompt not set. Call validate() first.")
        return self._system_prompt
