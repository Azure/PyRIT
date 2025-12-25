# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Manual script for evaluating multiple scorers against human-labeled datasets.

This is a long-running process that should be run occasionally to benchmark
scorer performance. Results are saved to the scorer_evals directory and checked in.

Usage:
    python build_scripts/evaluate_scorers.py
"""

import asyncio
import sys
from pathlib import Path

from pyrit.setup.initializers.simple import SimpleInitializer

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrit.common.path import SCORER_EVALS_PATH
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    SelfAskRefusalScorer,
    SelfAskLikertScorer,
)
from pyrit.setup import IN_MEMORY, initialize_pyrit_async


async def evaluate_scorers() -> None:
    """
    Evaluate multiple scorers against their configured datasets.
    
    This will:
    1. Initialize PyRIT with in-memory database
    2. Create a shared chat target for consistency
    3. Instantiate each scorer with appropriate configuration
    4. Run evaluate_async() on each scorer
    5. Save results to scorer_evals directory
    """
    print("Initializing PyRIT...")
    initializer = SimpleInitializer()
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    # Build list of scorers to evaluate
    scorers = [
        SelfAskRefusalScorer(chat_target=OpenAIChatTarget()),
        SimpleInitializer().get_default_objective_scorer(),
    ]
    
    print(f"\nEvaluating {len(scorers)} scorer(s)...\n")
    
    # Evaluate each scorer
    for i, scorer in enumerate(scorers, 1):
        scorer_name = scorer.__class__.__name__
        print(f"[{i}/{len(scorers)}] Evaluating {scorer_name}...")
        
        try:
            # Run evaluation with production settings:
            # - num_scorer_trials=3 for variance measurement
            # - add_to_evaluation_results=True to save to registry
            results = await scorer.evaluate_async(
                num_scorer_trials=3,
                add_to_evaluation_results=True,
                max_concurrency=10,
            )
            
            # Display results
            print(f"  ✓ Evaluation complete!")
            for dataset_name, metrics in results.items():
                print(f"    Dataset: {dataset_name}")
                print(f"      Accuracy: {metrics.accuracy:.2%} (±{metrics.accuracy_standard_error:.2%})")
                print(f"      Precision: {metrics.precision:.3f}")
                print(f"      Recall: {metrics.recall:.3f}")
                print(f"      F1 Score: {metrics.f1_score:.3f}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error evaluating {name}: {e}")
            print(f"    Continuing with next scorer...\n")
            continue
    
    print("=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {SCORER_EVALS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("PyRIT Scorer Evaluation Script")
    print("=" * 60)
    print("This script will evaluate multiple scorers against human-labeled")
    print("datasets. This is a long-running process that may take several")
    print("minutes to hours depending on the number of scorers and datasets.")
    print("=" * 60)
    print()
    
    try:
        asyncio.run(evaluate_scorers())
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)
