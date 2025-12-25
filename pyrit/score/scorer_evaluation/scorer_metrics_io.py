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

# Thread locks for writing (module-level, persists for application lifetime)
# Locks are created per file path to ensure thread-safe writes
_file_write_locks: Dict[str, threading.Lock] = {}


def load_all_metrics(file_path: Path) -> List[Dict]:
    """
    Load all scorer metrics entries from a JSONL file.
    
    Returns raw dictionaries for users who want to browse and compare scorer performance.
    Each dict contains scorer_identifier fields and metrics.
    
    Args:
        file_path (Path): Path to the JSONL file to load.
    
    Returns:
        List[Dict]: List of raw JSONL entries as dictionaries.
    """
    return _load_jsonl(file_path)


def find_metrics_by_hash(*, file_path: Path, hash: str, metrics_class: type) -> Optional["ScorerMetrics"]:
    """
    Find scorer metrics by configuration hash in a specific file.
    
    Args:
        file_path (Path): Path to the JSONL file to search.
        hash (str): The scorer configuration hash to search for.
        metrics_class (type): The metrics class to instantiate (ObjectiveScorerMetrics or HarmScorerMetrics).
    
    Returns:
        ScorerMetrics if found, else None.
    """
    entries = load_all_metrics(file_path)
    
    for entry in entries:
        if entry.get("hash") == hash:
            metrics_dict = entry.get("metrics", {})
            try:
                return metrics_class(**metrics_dict)
            except Exception as e:
                logger.warning(f"Failed to parse metrics for hash {hash}: {e}")
                return None
    
    return None


def add_evaluation_results(
    *,
    file_path: Path,
    scorer_identifier: ScorerIdentifier,
    metrics: "ScorerMetrics",
    dataset_version: str,
    harm_category: Optional[str] = None,
) -> None:
    """
    Append scorer metrics entry to the specified evaluation results file (thread-safe).
    
    This unified function handles both objective and harm scorer metrics, writing to
    the specified file path with appropriate validation and thread safety.
    
    Args:
        file_path (Path): The full path to the JSONL file to append to.
        scorer_identifier (ScorerIdentifier): The scorer's configuration identifier.
        metrics (ScorerMetrics): The computed metrics (ObjectiveScorerMetrics or HarmScorerMetrics).
        dataset_version (str): The version of the dataset used for evaluation.
        harm_category (Optional[str]): The harm category (required for HarmScorerMetrics).
    
    Raises:
        ValueError: If metrics is HarmScorerMetrics but harm_category is None.
    """
    from pyrit.score.scorer_evaluation.scorer_evaluator import HarmScorerMetrics, ObjectiveScorerMetrics
    
    # Validate harm_category for HarmScorerMetrics
    if isinstance(metrics, HarmScorerMetrics) and harm_category is None:
        raise ValueError("harm_category must be provided when metrics is HarmScorerMetrics")
    
    # Get or create lock for this file path
    file_path_str = str(file_path)
    if file_path_str not in _file_write_locks:
        _file_write_locks[file_path_str] = threading.Lock()
    
    # Build entry dictionary
    entry = scorer_identifier.to_compact_dict()
    
    if harm_category is not None:
        entry["harm_category"] = harm_category
    
    entry["metrics"] = asdict(metrics)
    
    # Write to file with thread safety
    _append_jsonl_entry(
        file_path=file_path,
        lock=_file_write_locks[file_path_str],
        entry=entry,
    )
    
    logger.info(f"Added metrics for {scorer_identifier.type} to {file_path.name}")


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
