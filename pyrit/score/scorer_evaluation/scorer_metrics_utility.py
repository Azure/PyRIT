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
from typing import Dict, List, Optional

from pyrit.common.path import SCORER_EVALS_HARM_PATH, SCORER_EVALS_OBJECTIVE_PATH
from pyrit.score.scorer_identifier import ScorerIdentifier

logger = logging.getLogger(__name__)

# File paths
OBJECTIVE_RESULTS_PATH = SCORER_EVALS_OBJECTIVE_PATH / "objective_evaluation_results.jsonl"

# Thread locks for writing (module-level, persists for application lifetime)
_objective_write_lock = threading.Lock()
_harm_write_locks: Dict[str, threading.Lock] = {}


def load_all_objective_metrics() -> List[Dict]:
    """
    Load all objective scorer metrics entries from JSONL file.
    
    Returns raw dictionaries for users who want to browse and compare scorer performance.
    Each dict contains scorer_identifier fields and metrics.
    
    Returns:
        List[Dict]: List of raw JSONL entries as dictionaries.
    """
    return _load_jsonl(OBJECTIVE_RESULTS_PATH)


def find_objective_metrics_by_hash(hash: str) -> Optional["ObjectiveScorerMetrics"]:
    """
    Find objective scorer metrics by configuration hash.
    
    Args:
        hash: The scorer configuration hash to search for.
    
    Returns:
        ObjectiveScorerMetrics if found, else None.
    """
    from pyrit.score.scorer_evaluation.scorer_evaluator import ObjectiveScorerMetrics
    
    entries = load_all_objective_metrics()
    
    for entry in entries:
        if entry.get("hash") == hash:
            metrics_dict = entry.get("metrics", {})
            try:
                return ObjectiveScorerMetrics(**metrics_dict)
            except Exception as e:
                logger.warning(f"Failed to parse metrics for hash {hash}: {e}")
                return None
    
    return None


def add_to_objective_evaluation_results(
    scorer_identifier: ScorerIdentifier,
    metrics: "ObjectiveScorerMetrics",
    dataset_version: str,
) -> None:
    """
    Append an objective scorer metrics entry to the evaluation results file (thread-safe).
    
    This should primarily be used by the PyRIT team when running evaluations on
    official datasets.
    
    Args:
        scorer_identifier: The scorer's configuration identifier.
        metrics: The computed objective metrics.
        dataset_version: The version of the dataset used for evaluation.
    """
    from pyrit.score.scorer_evaluation.scorer_evaluator import ObjectiveScorerMetrics
    
    entry = scorer_identifier.to_compact_dict()
    entry["dataset_version"] = dataset_version
    entry["metrics"] = asdict(metrics)
    
    _append_jsonl_entry(
        file_path=OBJECTIVE_RESULTS_PATH,
        lock=_objective_write_lock,
        entry=entry,
    )
    
    logger.info(f"Added objective metrics for {scorer_identifier.type} to evaluation results")


def add_to_harm_evaluation_results(
    scorer_identifier: ScorerIdentifier,
    metrics: "HarmScorerMetrics",
    harm_category: str,
    dataset_version: str,
) -> None:
    """
    Append a harm scorer metrics entry to the harm-specific evaluation results file (thread-safe).
    
    Each harm category has its own evaluation results file.
    
    Args:
        scorer_identifier: The scorer's configuration identifier.
        metrics: The computed harm metrics.
        harm_category: The harm category (e.g., "hate_speech", "violence").
        dataset_version: The version of the dataset used for evaluation.
    """
    from pyrit.score.scorer_evaluation.scorer_evaluator import HarmScorerMetrics
    
    # Get or create lock for this harm category
    if harm_category not in _harm_write_locks:
        _harm_write_locks[harm_category] = threading.Lock()
    
    harm_file_path = SCORER_EVALS_HARM_PATH / f"harm_{harm_category}_evaluation_results.jsonl"
    
    entry = scorer_identifier.to_compact_dict()
    entry["harm_category"] = harm_category
    entry["dataset_version"] = dataset_version
    entry["metrics"] = asdict(metrics)
    
    _append_jsonl_entry(
        file_path=harm_file_path,
        lock=_harm_write_locks[harm_category],
        entry=entry,
    )
    
    logger.info(f"Added harm metrics for {scorer_identifier.type} ({harm_category}) to evaluation results")


def load_harm_metrics(harm_category: str) -> List[Dict]:
    """
    Load all harm scorer metrics entries for a specific harm category.
    
    Args:
        harm_category: The harm category to load (e.g., "hate_speech", "violence").
    
    Returns:
        List[Dict]: List of raw JSONL entries as dictionaries.
    """
    harm_file_path = SCORER_EVALS_HARM_PATH / f"harm_{harm_category}_evaluation_results.jsonl"
    return _load_jsonl(harm_file_path)


def find_harm_metrics_by_hash(harm_category: str, hash: str) -> Optional["HarmScorerMetrics"]:
    """
    Find harm scorer metrics by configuration hash and harm category.
    
    Args:
        harm_category: The harm category to search in.
        hash: The scorer configuration hash to search for.
    
    Returns:
        HarmScorerMetrics if found, else None.
    """
    from pyrit.score.scorer_evaluation.scorer_evaluator import HarmScorerMetrics
    
    entries = load_harm_metrics(harm_category)
    
    for entry in entries:
        if entry.get("hash") == hash:
            metrics_dict = entry.get("metrics", {})
            try:
                return HarmScorerMetrics(**metrics_dict)
            except Exception as e:
                logger.warning(f"Failed to parse harm metrics for hash {hash}: {e}")
                return None
    
    return None


def _load_jsonl(file_path: Path) -> List[Dict]:
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


def _append_jsonl_entry(file_path: Path, lock: threading.Lock, entry: Dict) -> None:
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
