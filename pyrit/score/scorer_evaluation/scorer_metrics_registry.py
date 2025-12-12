# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pyrit
from pyrit.common.path import (
    SCORER_EVALS_HARM_PATH,
    SCORER_EVALS_OBJECTIVE_PATH,
)
from pyrit.common.singleton import Singleton

# Forward declaration for ScorerMetrics (imported from scorer_evaluator)
if TYPE_CHECKING:
    from pyrit.score.scorer_evaluation.scorer_evaluator import ScorerMetrics


class RegistryType(Enum):
    """Enum for different registry types."""

    HARM = "harm"
    OBJECTIVE = "objective"


@dataclass
class ScorerMetricsRegistryEntry:
    """
    A class that combines a ScorerEvalIdentifier with its evaluation metrics.

    This provides a clean interface for working with scorer evaluation results
    and encapsulates both the configuration and performance data.
    """

    scorer_identifier: "ScorerEvalIdentifier"
    metrics: "ScorerMetrics"
    dataset_version: str

    def get_accuracy(self) -> Optional[float]:
        """Get the accuracy metric if available."""
        return getattr(self.metrics, "accuracy", None)

    def get_mean_absolute_error(self) -> Optional[float]:
        """Get the mean absolute error metric if available."""
        return getattr(self.metrics, "mean_absolute_error", None)

    def print_summary(self) -> None:
        """Print a summary of the metrics."""
        self.scorer_identifier.print_summary()
        print("Metrics Summary:")
        print(json.dumps(asdict(self.metrics), indent=2))


@dataclass(frozen=True)
class ScorerEvalIdentifier:
    """
    Configuration class for Scorers.

    This class encapsulates the modifiable parameters that can be used to create a complete scoring configuration.
    These parameters can be modified, and configurations can be compared to each other via scorer evaluations.
    """

    type: str
    version: int
    system_prompt: Optional[str] = None
    sub_identifier: Optional[Union[Dict, List[Dict]]] = None
    model_info: Optional[Dict] = None
    scorer_specific_params: Optional[Dict] = None
    pyrit_version: str = pyrit.__version__

    def compute_hash(self) -> str:
        """
        Compute a hash representing the current configuration.

        Returns:
            str: A hash string representing the configuration.
        """
        import hashlib

        # Create a dictionary with all configuration parameters
        config_dict = {
            "type": self.type,
            "version": self.version,
            "sub_identifier": self.sub_identifier,
            "model_info": self.model_info,
            "system_prompt": self.system_prompt,
            "scorer_specific_params": self.scorer_specific_params,
            "pyrit_version": self.pyrit_version,
        }

        # Sort keys to ensure deterministic ordering and encode as JSON
        config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))

        hasher = hashlib.sha256()
        hasher.update(config_json.encode("utf-8"))
        return hasher.hexdigest()

    def print_summary(self) -> None:
        """Print a summary of the configuration."""
        print("ScorerEvalIdentifier Summary:")
        print(f"  Type: {self.type}")
        print(f"  Version: {self.version}")
        print(f"  Sub Identifier: {self.sub_identifier}")
        print(f"  Model Info: {self.model_info}")
        if self.system_prompt and len(self.system_prompt) > 100:
            prompt_display = self.system_prompt[:100] + "..."
        else:
            prompt_display = self.system_prompt
        print(f"  System Prompt: {prompt_display}")
        print(f"  Scorer Specific Params: {self.scorer_specific_params}")
        print(f"  PyRIT Version: {self.pyrit_version}")


class ScorerMetricsRegistry(metaclass=Singleton):
    """
    Registry for storing and retrieving scorer evaluation metrics.

    This class implements the singleton pattern to prevent race conditions
    when multiple instances try to modify the registry file.
    """

    # Class-level locks for each registry file to prevent race conditions
    _harm_file_lock: threading.Lock = threading.Lock()
    _objective_file_lock: threading.Lock = threading.Lock()

    # Map registry types to their corresponding locks
    _FILE_LOCKS = {RegistryType.HARM: _harm_file_lock, RegistryType.OBJECTIVE: _objective_file_lock}

    # File paths for different registry types
    _REGISTRY_FILES = {
        RegistryType.HARM: SCORER_EVALS_HARM_PATH / "harm_scorer_evals_registry.jsonl",
        RegistryType.OBJECTIVE: SCORER_EVALS_OBJECTIVE_PATH / "objective_scorer_evals_registry.jsonl",
    }

    def __init__(self) -> None:
        """
        Initialize the registry. Only runs once due to singleton pattern.
        """
        pass

    @property
    def harm_entries(self) -> List[dict]:
        """Get harm registry entries, always loading fresh from file."""
        return self._load_registry(RegistryType.HARM)

    @property
    def objective_entries(self) -> List[dict]:
        """Get objective registry entries, always loading fresh from file."""
        return self._load_registry(RegistryType.OBJECTIVE)

    def _get_file_path(self, registry_type: RegistryType) -> Path:
        """
        Get the file path for a specific registry type.

        Args:
            registry_type (RegistryType): The type of registry.

        Returns:
            Path: The file path for the registry.
        """
        return self._REGISTRY_FILES[registry_type]

    def _get_file_lock(self, registry_type: RegistryType) -> threading.Lock:
        """
        Get the file lock for a specific registry type.

        Args:
            registry_type (RegistryType): The type of registry.

        Returns:
            threading.Lock: The lock for the registry file.
        """
        return self._FILE_LOCKS[registry_type]

    def _load_registry(self, registry_type: RegistryType) -> List[dict]:
        """
        Load the registry from the JSONL file with thread safety.

        Args:
            registry_type (RegistryType): The type of registry to load.

        Returns:
            List[dict]: A list of registry entries.
        """
        file_path = self._get_file_path(registry_type)
        file_lock = self._get_file_lock(registry_type)

        with file_lock:
            entries: List[dict] = []

            # Create the file if it doesn't exist
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                return entries

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries

    def add_entry(
        self,
        *,
        scorer_identifier: ScorerEvalIdentifier,
        metrics: "ScorerMetrics",
        registry_type: RegistryType,
        dataset_version: str,
    ) -> None:
        """
        Add an entry to the specified registry with thread safety.

        Args:
            scorer_identifier (ScorerEvalIdentifier): The identifier of the scorer configuration.
            metrics (ScorerMetrics): The evaluation metrics to store.
            registry_type (RegistryType): The type of registry to add the entry to.
            dataset_version (str): The version of the dataset used for evaluation.
        """
        entry = {
            "hash": scorer_identifier.compute_hash(),
            "type": scorer_identifier.type,
            "version": scorer_identifier.version,
            "system_prompt": scorer_identifier.system_prompt,
            "sub_identifier": scorer_identifier.sub_identifier,
            "model_info": scorer_identifier.model_info,
            "scorer_specific_params": scorer_identifier.scorer_specific_params,
            "pyrit_version": scorer_identifier.pyrit_version,
            "dataset_version": dataset_version,
            "metrics": asdict(metrics),
        }

        file_path = self._get_file_path(registry_type)
        file_lock = self._get_file_lock(registry_type)

        with file_lock:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def get_scorer_registry_metrics_by_identifier(
        self, scorer_identifier: ScorerEvalIdentifier, registry_type: Optional[RegistryType] = None
    ) -> Optional["ScorerMetrics"]:
        hash = scorer_identifier.compute_hash()
        entry = self.get_metrics_registry_entries(registry_type=registry_type, hash=hash)
        if len(entry):
            return entry[0].metrics
        return None

    def get_metrics_registry_entries(
        self,
        *,
        registry_type: Optional[RegistryType] = None,
        hash: Optional[str] = None,
        type: Optional[str] = None,
        version: Optional[int] = None,
        dataset_version: Optional[str] = None,
        # Model info filters
        model_name: Optional[str] = None,
        model_temperature: Optional[float] = None,
        model_top_p: Optional[float] = None,
        target_name: Optional[str] = None,
        target_metadata: Optional[Dict] = None,
        # System prompt filter
        system_prompt: Optional[str] = None,
        # Scorer specific params filter
        scorer_specific_params: Optional[Dict] = None,
        # General filters
        pyrit_version: Optional[str] = None,
        # Metrics threshold filters
        accuracy_threshold: Optional[float] = None,
        mean_absolute_error_threshold: Optional[float] = None,
    ) -> List[ScorerMetricsRegistryEntry]:
        """
        Retrieves a list of ScorerMetricsRegistryEntry objects based on the specified filters.
        Results are ordered by accuracy from highest to lowest.

        Args:
            registry_type (Optional[RegistryType]): The type of registry to query.
                If None, searches both registries.
            hash (Optional[str]): The hash to filter by. Defaults to None.
            type (Optional[str]): The scorer type to filter by. Defaults to None.
            version (Optional[int]): The version to filter by. Defaults to None.
            dataset_version (Optional[str]): The dataset version to filter by. Defaults to None.
            model_name (Optional[str]): The model name in model_info to filter by. Defaults to None.
            model_temperature (Optional[float]): The model temperature in model_info to filter by. Defaults to None.
            model_top_p (Optional[float]): The model top_p in model_info to filter by. Defaults to None.
            target_name (Optional[str]): The target name in model_info to filter by. Defaults to None.
            target_metadata (Optional[Dict]): The target metadata in model_info to filter by. Defaults to None.
            system_prompt (Optional[str]): The system prompt to filter by. Defaults to None.
            scorer_specific_params (Optional[Dict]): The scorer specific parameters to filter by. Defaults to None.
            pyrit_version (Optional[str]): The PyRIT version to filter by. Defaults to None.
            accuracy_threshold (Optional[float]): Minimum accuracy threshold for metrics. Defaults to None.
            mean_absolute_error_threshold (Optional[float]): Maximum mean absolute error threshold for metrics. Defaults to None.

        Returns:
            List[ScorerMetricsRegistryEntry]: A list of ScorerMetricsRegistryEntry objects that match the specified filters,
                ordered by accuracy from highest to lowest.
        """
        # Import here to avoid circular import
        from pyrit.score.scorer_evaluation.scorer_evaluator import (
            HarmScorerMetrics,
            ObjectiveScorerMetrics,
        )

        # Determine which entries to search
        entries = []
        if registry_type is None:
            # Search both registries
            entries.extend(self.harm_entries)
            entries.extend(self.objective_entries)
        elif registry_type == RegistryType.HARM:
            entries = self.harm_entries
        elif registry_type == RegistryType.OBJECTIVE:
            entries = self.objective_entries

        filtered_entries: List[ScorerMetricsRegistryEntry] = []

        for entry in entries:
            # Basic field filters
            if hash and entry.get("hash") != hash:
                continue
            if type and entry.get("type") != type:
                continue
            if version and entry.get("version") != version:
                continue
            if dataset_version and entry.get("dataset_version") != dataset_version:
                continue
            if pyrit_version and entry.get("pyrit_version") != pyrit_version:
                continue
            if system_prompt and entry.get("system_prompt") != system_prompt:
                continue
            if scorer_specific_params and entry.get("scorer_specific_params") != scorer_specific_params:
                continue

            # Nested model_info filters
            model_info = entry.get("model_info", {})
            if model_name and model_info.get("model_name") != model_name:
                continue
            if model_temperature is not None and model_info.get("temperature") != model_temperature:
                continue
            if model_top_p is not None and model_info.get("top_p") != model_top_p:
                continue
            if target_name and model_info.get("target_name") != target_name:
                continue
            if target_metadata and model_info.get("custom_metadata") != target_metadata:
                continue

            # Metrics threshold filters
            metrics = entry.get("metrics", {})
            if accuracy_threshold is not None and metrics.get("accuracy", 0) < accuracy_threshold:
                continue
            if (
                mean_absolute_error_threshold is not None
                and metrics.get("mean_absolute_error", float("inf")) > mean_absolute_error_threshold
            ):
                continue

            # If we made it here, the entry passes all filters
            scorer_identifier = ScorerEvalIdentifier(
                type=entry.get("type"),
                version=entry.get("version"),
                sub_identifier=entry.get("sub_identifier", []),
                model_info=entry.get("model_info"),
                system_prompt=entry.get("system_prompt"),
                scorer_specific_params=entry.get("scorer_specific_params"),
                pyrit_version=entry.get("pyrit_version"),
            )

            # Reconstruct the appropriate ScorerMetrics object from stored dict
            metrics_dict = entry.get("metrics", {})

            # Determine metrics type based on available fields
            if "accuracy" in metrics_dict:
                metrics = ObjectiveScorerMetrics(**metrics_dict)
            else:
                metrics = HarmScorerMetrics(**metrics_dict)

            metrics_entry = ScorerMetricsRegistryEntry(
                scorer_identifier=scorer_identifier, metrics=metrics, dataset_version=entry.get("dataset_version")
            )
            filtered_entries.append(metrics_entry)

        # Sort filtered entries by accuracy (highest to lowest)
        if registry_type == RegistryType.OBJECTIVE:
            filtered_entries.sort(key=lambda entry: entry.get_accuracy() or 0.0, reverse=True)
        elif registry_type == RegistryType.HARM:
            filtered_entries.sort(
                key=lambda entry: entry.get_mean_absolute_error() or float("inf"),
            )

        return filtered_entries
