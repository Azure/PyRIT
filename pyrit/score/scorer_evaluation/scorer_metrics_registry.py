# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import threading
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

from pyrit.common.path import (
    SCORER_EVALS_HARM_PATH,
    SCORER_EVALS_OBJECTIVE_PATH,
)
from pyrit.common.singleton import Singleton
from pyrit.score.scorer_identifier import ScorerIdentifier

# Forward declaration for ScorerMetrics (imported from scorer_evaluator)
if TYPE_CHECKING:
    from pyrit.score.scorer_evaluation.scorer_evaluator import ScorerMetrics


class RegistryType(Enum):
    """Enum for different registry types."""

    HARM = "harm"
    OBJECTIVE = "objective"


class ScorerMetricsEntry(NamedTuple):
    """
    A lightweight container pairing a scorer's configuration with its evaluation metrics.

    This allows callers to know which specific scorer configuration produced each set of metrics
    when filtering returns multiple results.
    """

    scorer_identifier: Dict[str, Any]
    """The scorer configuration (type, version, prompts, model_info, etc.)"""

    metrics: "ScorerMetrics"
    """The evaluation metrics (accuracy, precision, recall, etc.)"""


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
        scorer_identifier: ScorerIdentifier,
        metrics: "ScorerMetrics",
        registry_type: RegistryType,
        dataset_version: str,
    ) -> None:
        """
        Add an entry to the specified registry with thread safety.

        Args:
            scorer_identifier (ScorerIdentifier): The identifier of the scorer configuration.
            metrics (ScorerMetrics): The evaluation metrics to store.
            registry_type (RegistryType): The type of registry to add the entry to.
            dataset_version (str): The version of the dataset used for evaluation.
        """
        # Use to_compact_dict() for consistent serialization with __type__ and compacted prompts
        entry = scorer_identifier.to_compact_dict()
        entry["hash"] = scorer_identifier.compute_hash()
        entry["dataset_version"] = dataset_version
        entry["metrics"] = asdict(metrics)

        file_path = self._get_file_path(registry_type)
        file_lock = self._get_file_lock(registry_type)

        with file_lock:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def get_scorer_registry_metrics_by_identifier(
        self, scorer_identifier: ScorerIdentifier, registry_type: Optional[RegistryType] = None
    ) -> Optional["ScorerMetrics"]:
        """
        Get the evaluation metrics for a specific scorer identifier.

        Args:
            scorer_identifier (ScorerIdentifier): The identifier of the scorer configuration.
            registry_type (Optional[RegistryType]): The type of registry to search. If None, searches both.

        Returns:
            Optional[ScorerMetrics]: The evaluation metrics if found, else None.
        """
        hash = scorer_identifier.compute_hash()
        entries = self.get_metrics_registry_entries(registry_type=registry_type, hash=hash)
        if len(entries):
            return entries[0].metrics
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
        # System prompt template filter
        system_prompt_template: Optional[str] = None,
        # Scorer specific params filter
        scorer_specific_params: Optional[Dict] = None,
        # General filters
        pyrit_version: Optional[str] = None,
        # Metrics threshold filters
        accuracy_threshold: Optional[float] = None,
        mean_absolute_error_threshold: Optional[float] = None,
    ) -> List[ScorerMetricsEntry]:
        """
        Retrieve a list of ScorerMetricsEntry objects based on the specified filters.
        Each entry contains the scorer configuration and its evaluation metrics.
        Results are ordered by accuracy (highest to lowest) for OBJECTIVE registry,
        or by mean absolute error (lowest to highest) for HARM registry.

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
            system_prompt_template (Optional[str]): The system prompt template to filter by. Defaults to None.
            scorer_specific_params (Optional[Dict]): The scorer specific parameters to filter by. Defaults to None.
            pyrit_version (Optional[str]): The PyRIT version to filter by. Defaults to None.
            accuracy_threshold (Optional[float]): Minimum accuracy threshold for metrics. Defaults to None.
            mean_absolute_error_threshold (Optional[float]): Maximum mean absolute error threshold for metrics.
                Defaults to None.

        Returns:
            List[ScorerMetricsEntry]: A list of (scorer_identifier, metrics) pairs that match the filters.
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

        filtered_entries: List[ScorerMetricsEntry] = []

        for entry in entries:
            # Basic field filters
            if hash and entry.get("hash") != hash:
                continue
            if type and entry.get("__type__") != type:
                continue
            if version and entry.get("version") != version:
                continue
            if dataset_version and entry.get("dataset_version") != dataset_version:
                continue
            if pyrit_version and entry.get("pyrit_version") != pyrit_version:
                continue
            if system_prompt_template and entry.get("system_prompt_template") != system_prompt_template:
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
            metrics_dict = entry.get("metrics", {})
            if accuracy_threshold is not None and metrics_dict.get("accuracy", 0) < accuracy_threshold:
                continue
            if (
                mean_absolute_error_threshold is not None
                and metrics_dict.get("mean_absolute_error", float("inf")) > mean_absolute_error_threshold
            ):
                continue

            # If we made it here, the entry passes all filters
            # Build scorer_identifier dict (everything except metrics)
            scorer_identifier = {k: v for k, v in entry.items() if k != "metrics"}

            # Reconstruct the appropriate ScorerMetrics object from stored dict
            metrics: "ScorerMetrics"
            if "accuracy" in metrics_dict:
                metrics = ObjectiveScorerMetrics(**metrics_dict)
            else:
                metrics = HarmScorerMetrics(**metrics_dict)

            filtered_entries.append(ScorerMetricsEntry(scorer_identifier=scorer_identifier, metrics=metrics))

        # Sort filtered entries
        if registry_type == RegistryType.OBJECTIVE:
            filtered_entries.sort(key=lambda e: getattr(e.metrics, "accuracy", 0.0), reverse=True)
        elif registry_type == RegistryType.HARM:
            filtered_entries.sort(key=lambda e: getattr(e.metrics, "mean_absolute_error", float("inf")))

        return filtered_entries
