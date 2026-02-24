# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for reading/writing scorer evaluation metrics to JSONL files.
Thread-safe operations for appending entries.
"""

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pyrit.common.path import (
    SCORER_EVALS_PATH,
)
from pyrit.identifiers import ComponentIdentifier, config_hash
from pyrit.score.scorer_evaluation.scorer_metrics import (
    HarmScorerMetrics,
    ObjectiveScorerMetrics,
    ScorerMetrics,
    ScorerMetricsWithIdentity,
)

logger = logging.getLogger(__name__)

# Thread locks for writing (module-level, persists for application lifetime)
# Locks are created per file path to ensure thread-safe writes
_file_write_locks: Dict[str, threading.Lock] = {}

M = TypeVar("M", bound=ScorerMetrics)

# Child component params that affect scoring behavior.
# Operational params (endpoint, max_requests_per_minute, etc.) are excluded
# so that the same model on different deployments shares cached eval results.
_BEHAVIORAL_CHILD_PARAMS = frozenset({"model_name", "temperature", "top_p"})
_TARGET_CHILD_KEYS = frozenset({"prompt_target", "converter_target"})


def _build_eval_dict(
    identifier: ComponentIdentifier,
    *,
    param_allowlist: Optional[frozenset[str]] = None,
) -> Dict[str, Any]:
    """
    Build a dictionary for eval hashing.

    This function creates a filtered representation of a component's configuration,
    including only behavioral parameters. For child components that are targets,
    only behavioral params are included. For non-target children, full evaluation
    treatment is applied recursively.

    Args:
        identifier (ComponentIdentifier): The component identity to process.
        param_allowlist (Optional[frozenset[str]]): If provided, only include
            params whose keys are in the allowlist. If None, include all params.
            Target children are filtered to _BEHAVIORAL_CHILD_PARAMS, while
            non-target children receive full eval treatment without param filtering.

    Returns:
        Dict[str, Any]: The filtered dictionary suitable for hashing.
    """
    eval_dict: Dict[str, Any] = {
        ComponentIdentifier.KEY_CLASS_NAME: identifier.class_name,
        ComponentIdentifier.KEY_CLASS_MODULE: identifier.class_module,
    }

    for key, value in sorted(identifier.params.items()):
        if value is not None and (param_allowlist is None or key in param_allowlist):
            eval_dict[key] = value

    if identifier.children:
        eval_children: Dict[str, Any] = {}
        for name in sorted(identifier.children):
            child_list = identifier.get_child_list(name)
            if name in _TARGET_CHILD_KEYS:
                # Targets: filter to behavioral params only
                hashes = [
                    config_hash(_build_eval_dict(c, param_allowlist=_BEHAVIORAL_CHILD_PARAMS)) for c in child_list
                ]
            else:
                # Non-targets (e.g., sub-scorers): full eval treatment, recurse without param filtering
                hashes = [config_hash(_build_eval_dict(c)) for c in child_list]
            eval_children[name] = hashes[0] if len(hashes) == 1 else hashes
        if eval_children:
            eval_dict["children"] = eval_children

    return eval_dict


def compute_eval_hash(identifier: ComponentIdentifier) -> str:
    """
    Compute a behavioral equivalence hash for scorer evaluation grouping.

    Includes all of the scorer's own params but projects child components
    down to only behavioral params (model_name, temperature, top_p).

    Args:
        identifier (ComponentIdentifier): The scorer's full identity.

    Returns:
        str: A hash suitable for eval registry keying.
    """
    return config_hash(_build_eval_dict(identifier))


def _metrics_to_registry_dict(metrics: ScorerMetrics) -> Dict[str, Any]:
    """
    Convert metrics to a dictionary suitable for registry storage.

    Excludes:
    - trial_scores (too large for registry storage)
    - Internal fields starting with '_'

    Args:
        metrics (ScorerMetrics): The metrics object to convert.

    Returns:
        Dict: A dictionary with excluded fields removed.
    """
    metrics_dict = asdict(metrics)
    excluded_keys = {"trial_scores"}
    return {k: v for k, v in metrics_dict.items() if k not in excluded_keys and v is not None and not k.startswith("_")}


def get_all_objective_metrics(
    file_path: Optional[Path] = None,
) -> List[ScorerMetricsWithIdentity[ObjectiveScorerMetrics]]:
    """
    Load all objective scorer metrics with full scorer identity for comparison.

    Returns a list of ScorerMetricsWithIdentity[ObjectiveScorerMetrics] objects that wrap
    the scorer's identity information and its performance metrics, enabling clean attribute
    access like `entry.metrics.accuracy` or `entry.metrics.f1_score`.

    Args:
        file_path (Optional[Path]): Path to a specific JSONL file to load.
            If not provided, uses the default path:
            SCORER_EVALS_PATH / "objective" / "objective_achieved_metrics.jsonl"

    Returns:
        List[ScorerMetricsWithIdentity[ObjectiveScorerMetrics]]: List of metrics with scorer identity.
            Access metrics via `entry.metrics.accuracy`, `entry.metrics.f1_score`, etc.
            Access scorer info via `entry.scorer_identifier.class_name`, etc.
    """
    if file_path is None:
        file_path = SCORER_EVALS_PATH / "objective" / "objective_achieved_metrics.jsonl"

    return _load_metrics_from_file(file_path=file_path, metrics_class=ObjectiveScorerMetrics)


def get_all_harm_metrics(
    harm_category: str,
) -> List[ScorerMetricsWithIdentity[HarmScorerMetrics]]:
    """
    Load all harm scorer metrics for a specific harm category.

    Returns a list of ScorerMetricsWithIdentity[HarmScorerMetrics] objects that wrap
    the scorer's identity information and its performance metrics, enabling clean attribute
    access like `entry.metrics.mean_absolute_error` or `entry.metrics.harm_category`.

    Args:
        harm_category (str): The harm category to load metrics for (e.g., "hate_speech", "violence").

    Returns:
        List[ScorerMetricsWithIdentity[HarmScorerMetrics]]: List of metrics with scorer identity.
            Access metrics via `entry.metrics.mean_absolute_error`, `entry.metrics.harm_category`, etc.
            Access scorer info via `entry.scorer_identifier.class_name`, etc.
    """
    file_path = SCORER_EVALS_PATH / "harm" / f"{harm_category}_metrics.jsonl"
    return _load_metrics_from_file(file_path=file_path, metrics_class=HarmScorerMetrics)


def _load_metrics_from_file(
    *,
    file_path: Path,
    metrics_class: Type[M],
) -> List[ScorerMetricsWithIdentity[M]]:
    """
    Load scorer metrics from a JSONL file with the specified metrics class.

    This is a private helper function used by get_all_objective_metrics and get_all_harm_metrics.

    Args:
        file_path (Path): Path to the JSONL file to load.
        metrics_class (Type[M]): The metrics class to instantiate (ObjectiveScorerMetrics or HarmScorerMetrics).

    Returns:
        List[ScorerMetricsWithIdentity[M]]: List of metrics with scorer identity.
    """
    results: List[ScorerMetricsWithIdentity[M]] = []
    entries = _load_jsonl(file_path)

    for entry in entries:
        metrics_dict = entry.get("metrics", {})
        # Filter out internal fields that have init=False (e.g., _harm_definition_obj)
        metrics_dict = {k: v for k, v in metrics_dict.items() if not k.startswith("_")}

        # Extract scorer identity (everything except metrics)
        identity_dict = {k: v for k, v in entry.items() if k not in ("metrics", "eval_hash")}

        try:
            # Reconstruct ComponentIdentifier from the stored dict
            scorer_identifier = ComponentIdentifier.from_dict(identity_dict)

            # Create the metrics object
            metrics = metrics_class(**metrics_dict)

            results.append(
                ScorerMetricsWithIdentity(
                    scorer_identifier=scorer_identifier,
                    metrics=metrics,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse metrics entry: {e}")
            continue

    return results


def find_objective_metrics_by_hash(
    *,
    hash: str,
    file_path: Optional[Path] = None,
) -> Optional[ObjectiveScorerMetrics]:
    """
    Find objective scorer metrics by configuration hash.

    Args:
        hash (str): The scorer configuration hash to search for.
        file_path (Optional[Path]): Path to the JSONL file to search.
            If not provided, uses the default path:
            SCORER_EVALS_PATH / "objective" / "objective_achieved_metrics.jsonl"

    Returns:
        ObjectiveScorerMetrics if found, else None.
    """
    if file_path is None:
        file_path = SCORER_EVALS_PATH / "objective" / "objective_achieved_metrics.jsonl"

    return _find_metrics_by_hash(file_path=file_path, hash=hash, metrics_class=ObjectiveScorerMetrics)


def find_harm_metrics_by_hash(
    *,
    hash: str,
    harm_category: str,
) -> Optional[HarmScorerMetrics]:
    """
    Find harm scorer metrics by configuration hash.

    Args:
        hash (str): The scorer configuration hash to search for.
        harm_category (str): The harm category to search in (e.g., "hate_speech", "violence").

    Returns:
        HarmScorerMetrics if found, else None.
    """
    file_path = SCORER_EVALS_PATH / "harm" / f"{harm_category}_metrics.jsonl"
    return _find_metrics_by_hash(file_path=file_path, hash=hash, metrics_class=HarmScorerMetrics)


def _find_metrics_by_hash(
    *,
    file_path: Path,
    hash: str,
    metrics_class: Type[M],
) -> Optional[M]:
    """
    Find scorer metrics by configuration hash in a specific file.

    This is a private helper function used by find_objective_metrics_by_hash and find_harm_metrics_by_hash.

    Args:
        file_path (Path): Path to the JSONL file to search.
        hash (str): The scorer configuration hash to search for.
        metrics_class (Type[M]): The metrics class to instantiate.

    Returns:
        The metrics instance if found, else None.
    """
    entries = _load_jsonl(file_path)

    for entry in entries:
        if entry.get("hash") == hash:
            metrics_dict = entry.get("metrics", {})
            # Filter out internal fields that have init=False (e.g., _harm_definition_obj)
            metrics_dict = {k: v for k, v in metrics_dict.items() if not k.startswith("_")}
            try:
                return metrics_class(**metrics_dict)
            except Exception as e:
                logger.warning(f"Failed to parse metrics for hash {hash}: {e}")
                return None

    return None


def add_evaluation_results(
    *,
    file_path: Path,
    scorer_identifier: ComponentIdentifier,
    metrics: "ScorerMetrics",
) -> None:
    """
    Append scorer metrics entry to the specified evaluation results file (thread-safe).

    This unified function handles both objective and harm scorer metrics, writing to
    the specified file path with appropriate validation and thread safety.

    Args:
        file_path (Path): The full path to the JSONL file to append to.
        scorer_identifier (ComponentIdentifier): The scorer's configuration identifier.
        metrics (ScorerMetrics): The computed metrics (ObjectiveScorerMetrics or HarmScorerMetrics).
    """
    # Get or create lock for this file path
    file_path_str = str(file_path)
    if file_path_str not in _file_write_locks:
        _file_write_locks[file_path_str] = threading.Lock()

    eval_hash = compute_eval_hash(scorer_identifier)

    # Build entry dictionary
    entry = scorer_identifier.to_dict()
    entry["eval_hash"] = eval_hash
    entry["metrics"] = _metrics_to_registry_dict(metrics)

    # Write to file with thread safety
    _append_jsonl_entry(
        file_path=file_path,
        lock=_file_write_locks[file_path_str],
        entry=entry,
    )

    logger.info(f"Added metrics for {scorer_identifier.class_name} to {file_path.name}")


def _load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load entries from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.
    """
    if not file_path.exists():
        logger.debug(f"Registry file not found: {file_path}")
        return []

    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num} in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to load registry from {file_path}: {e}")

    return entries


def _append_jsonl_entry(file_path: Path, lock: threading.Lock, entry: Dict[str, Any]) -> None:
    """
    Append an entry to a JSONL file with thread safety.

    Args:
        file_path: Path to the JSONL file.
        lock: Threading lock to ensure atomic writes.
        entry: Dictionary to append as JSON line.
    """
    with lock:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to registry {file_path}: {e}")
            raise


def replace_evaluation_results(
    *,
    file_path: Path,
    scorer_identifier: ComponentIdentifier,
    metrics: "ScorerMetrics",
) -> None:
    """
    Replace existing scorer metrics entry (by hash) with new metrics, or add if not exists.

    This is an atomic operation that removes any existing entry with the same scorer hash
    and adds the new entry. Only one entry per scorer hash is maintained in the registry,
    ensuring we always track the highest-fidelity evaluation.

    Args:
        file_path (Path): The full path to the JSONL file.
        scorer_identifier (ComponentIdentifier): The scorer's configuration identifier.
        metrics (ScorerMetrics): The computed metrics (ObjectiveScorerMetrics or HarmScorerMetrics).
    """
    # Get or create lock for this file path
    file_path_str = str(file_path)
    if file_path_str not in _file_write_locks:
        _file_write_locks[file_path_str] = threading.Lock()

    eval_hash = compute_eval_hash(scorer_identifier)

    # Build new entry dictionary
    new_entry = scorer_identifier.to_dict()
    new_entry["eval_hash"] = eval_hash
    new_entry["metrics"] = _metrics_to_registry_dict(metrics)

    with _file_write_locks[file_path_str]:
        try:
            # Load existing entries
            existing_entries = _load_jsonl(file_path)

            # Filter out entries with the same hash
            filtered_entries = [e for e in existing_entries if e.get("eval_hash") != eval_hash]

            # Add the new entry
            filtered_entries.append(new_entry)

            # Rewrite the file atomically
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                for entry in filtered_entries:
                    f.write(json.dumps(entry) + "\n")

            replaced = len(existing_entries) != len(filtered_entries)
            action = "Replaced" if replaced else "Added"
            logger.info(
                f"{action} metrics for {scorer_identifier.class_name} (eval_hash={eval_hash[:8]}...) in {file_path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to replace entry in registry {file_path}: {e}")
            raise
