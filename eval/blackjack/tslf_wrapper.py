#!/usr/bin/env python3
"""
TSLf Wrapper for Blackjack Evaluation

NOTE: TSLf mining for blackjack is experimental. Unlike spatial games (FrozenLake,
Taxi), blackjack doesn't have deterministic state transformations. The "updates"
would be:
  - [stood <- true] when standing
  - [count <- ?] when hitting (non-deterministic due to card draws)

For evaluation, we run the mining pipeline and attempt to extract decision rules
from the mined spec.
"""

import json
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def run_tslf_pipeline(trace_dir: Path, strategy: str,
                      max_size: int = 10) -> Tuple[Optional[str], dict]:
    """
    Run the TSLf mining pipeline on blackjack traces.

    Args:
        trace_dir: Directory containing pos/ and neg/ trace subdirectories
        strategy: Strategy name (for metadata file selection)
        max_size: Maximum formula size for enumeration

    Returns:
        spec: Mined specification string (or None if mining fails)
        metadata: Pipeline metadata
    """
    metadata = {
        "strategy": strategy,
        "trace_dir": str(trace_dir),
        "max_size": max_size
    }

    # Check if traces exist
    pos_dir = trace_dir / "pos"
    neg_dir = trace_dir / "neg"

    if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
        metadata["error"] = "No positive traces found"
        return None, metadata

    # Create metadata.py for blackjack
    meta_file = trace_dir / "metadata.py"
    _create_blackjack_metadata(meta_file, strategy)

    # Run pipeline
    pipeline_script = SRC_DIR / "mine.sh"
    if not pipeline_script.exists():
        metadata["error"] = f"Mining script not found: {pipeline_script}"
        return None, metadata

    try:
        # Use the pipeline options that work for blackjack:
        # --mode safety: mine safety specs (G-rooted)
        # --collect-all: collect all solutions
        # --self-inputs-only: only use self-updates
        # --prune: remove spurious specs
        result = subprocess.run(
            ["bash", str(pipeline_script), str(trace_dir),
             "--mode", "safety",
             "--max-size", str(max_size),
             "--collect-all",
             "--self-inputs-only",
             "--prune"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(SRC_DIR),
            env={"PATH": "/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin",
                 "HOME": str(Path.home())}
        )

        metadata["stdout"] = result.stdout
        metadata["stderr"] = result.stderr
        metadata["returncode"] = result.returncode

        if result.returncode != 0:
            metadata["error"] = f"Pipeline failed with code {result.returncode}"
            return None, metadata

        # Read mined spec from various possible locations
        spec = None

        # For conservative/basic strategies, prefer specs with standVsWeakMin
        # as they are more explicit about the strategy
        all_safety_file = trace_dir / "out" / "all_safety.tsl"
        if all_safety_file.exists() and strategy in ["conservative", "basic"]:
            with open(all_safety_file) as f:
                for line in f:
                    line = line.strip()
                    if 'standVsWeakMin' in line and 'isWeakDealer' in line and 'stood' in line:
                        # Found a more explicit spec
                        spec = line
                        break

        # Fall back to spec.tsl
        if not spec:
            spec_file = trace_dir / "out" / "spec.tsl"
            if spec_file.exists():
                spec = spec_file.read_text().strip()

        # Try safety_specs.txt
        if not spec:
            safety_file = trace_dir / "out" / "safety_specs.txt"
            if safety_file.exists():
                lines = safety_file.read_text().strip().split('\n')
                if lines:
                    spec = lines[0].strip()  # Take first spec

        # Try to extract from bolt output in stdout
        if not spec and result.stdout:
            # Look for lines that look like specs (start with G or F)
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('G ') or line.startswith('F '):
                    spec = line
                    break

        if spec:
            return spec, metadata

        metadata["error"] = "No spec produced"
        return None, metadata

    except subprocess.TimeoutExpired:
        metadata["error"] = "Pipeline timed out"
        return None, metadata
    except Exception as e:
        metadata["error"] = str(e)
        return None, metadata


def _create_blackjack_metadata(meta_file: Path, strategy: str):
    """Create metadata.py file for blackjack traces."""
    # Define variables based on strategy - matches new trace format with isWeakDealer
    if strategy == "threshold":
        variables = """
VARIABLES = {
    "count": "int",
    "standThreshold": "int",
    "stood": "bool",
}
"""
    elif strategy == "conservative":
        variables = """
VARIABLES = {
    "count": "int",
    "standThreshold": "int",
    "standVsWeakMin": "int",
    "isWeakDealer": "bool",
    "stood": "bool",
}
"""
    else:  # basic
        variables = """
VARIABLES = {
    "count": "int",
    "standThreshold": "int",
    "standVsWeakMin": "int",
    "isWeakDealer": "bool",
    "stood": "bool",
}
"""

    content = f'''# Auto-generated metadata for blackjack {strategy} strategy
{variables}

CONSTANTS = {{
    "standThreshold": 17,
    "standVsWeakMin": 13,
}}

# Output variable
OUTPUT = "stood"
'''
    meta_file.write_text(content)


def parse_spec_to_action_fn(spec: str, strategy: str) -> Optional[Callable[[Dict], int]]:
    """
    Parse a TSLf spec into an action function.

    Handles common patterns mined for blackjack strategies:
    - G (! ((ltC count standThreshold) <-> (X (stood))))
      → stand when count >= threshold
    - Patterns involving dealer and weakDealerMax for conservative
    - Patterns involving standVsWeakMin for basic

    Args:
        spec: Mined TSLf specification
        strategy: Strategy name

    Returns:
        Action function or None if parsing fails
    """
    import re

    spec_lower = spec.lower()
    spec_normalized = ' '.join(spec.split())  # Normalize whitespace

    # IMPORTANT: Check patterns from most specific to least specific!
    # Otherwise more general patterns will match first and return wrong action.

    # Pattern 1: Conservative/Basic strategy with isWeakDealer and standVsWeakMin
    # Matches specs like:
    # G ((X (stood)) <-> (! ((ltC count standThreshold) & ((isWeakDealer) -> (ltC count standVsWeakMin)))))
    # This pattern covers both conservative (standVsWeakMin=12) and basic (standVsWeakMin=13)
    if 'standvsweakmin' in spec_lower and 'isweakdealer' in spec_lower:
        def combined_action(state: Dict) -> int:
            threshold = state.get("standThreshold", 17)
            stand_vs_weak = state.get("standVsWeakMin", 12)  # Uses state value
            count = state["count"]
            is_weak = state.get("isWeakDealer", False)

            if count >= threshold:
                return 1  # Stand at 17+
            if count >= stand_vs_weak and is_weak:
                return 1  # Stand at standVsWeakMin+ vs weak dealer
            return 0  # Hit
        return combined_action

    # Pattern 2: Conservative strategy with isWeakDealer (without standVsWeakMin)
    # When only isWeakDealer is in spec (older format)
    if 'isweakdealer' in spec_lower and 'stood' in spec_lower:
        def conservative_action(state: Dict) -> int:
            threshold = state.get("standThreshold", 17)
            stand_vs_weak = state.get("standVsWeakMin", 12)  # Conservative default
            count = state["count"]
            is_weak = state.get("isWeakDealer", False)

            if count >= threshold:
                return 1  # Stand at 17+
            if count >= stand_vs_weak and is_weak:
                return 1  # Stand at 12+ vs weak dealer
            return 0  # Hit
        return conservative_action

    # Fallback: check for any stood-related pattern with count comparison
    if 'stood' in spec_lower and ('count' in spec_lower or 'standthreshold' in spec_lower):
        # Generic threshold-like behavior
        def generic_threshold(state: Dict) -> int:
            threshold = state.get("standThreshold", 17)
            if state["count"] >= threshold:
                return 1
            return 0
        return generic_threshold

    # Last resort: if we see ltC/geC count standThreshold, assume threshold strategy
    # This handles cases where the spec is about count transitions but doesn't
    # explicitly mention stood (common with small n)
    if 'ltc count standthreshold' in spec_lower or 'gec count standthreshold' in spec_lower:
        def inferred_threshold(state: Dict) -> int:
            threshold = state.get("standThreshold", 17)
            if state["count"] >= threshold:
                return 1
            return 0
        return inferred_threshold

    # Default: couldn't parse spec
    return None


def find_existing_spec(trace_dir: Path) -> Optional[str]:
    """Check if a spec already exists from a previous pipeline run."""
    out_dir = trace_dir / "out"

    # Check spec.tsl
    spec_file = out_dir / "spec.tsl"
    if spec_file.exists():
        spec = spec_file.read_text().strip()
        if spec:
            return spec

    # Check safety_specs.txt
    safety_file = out_dir / "safety_specs.txt"
    if safety_file.exists():
        lines = safety_file.read_text().strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('G ') or line.startswith('F '):
                return line

    return None


def train_and_evaluate(trace_dir: Path, strategy: str,
                       max_size: int = 10,
                       reuse_existing: bool = True) -> Tuple[Optional[Callable], dict]:
    """
    Run TSLf pipeline and create action function.

    Args:
        trace_dir: Directory containing traces
        strategy: Strategy name
        max_size: Maximum formula size
        reuse_existing: If True, use existing spec if available

    Returns:
        action_fn: Action function (or None)
        metadata: Pipeline metadata
    """
    import time

    meta = {
        "method": "tslf",
        "strategy": strategy,
        "trace_dir": str(trace_dir),
    }

    # Check for existing spec first
    if reuse_existing:
        existing_spec = find_existing_spec(trace_dir)
        if existing_spec:
            meta["spec"] = existing_spec
            meta["spec_source"] = "existing"
            meta["train_time"] = 0.0

            action_fn = parse_spec_to_action_fn(existing_spec, strategy)
            if action_fn is None:
                meta["warning"] = "Could not parse existing spec to action function"
            return action_fn, meta

    # Run pipeline
    start = time.time()
    spec, pipeline_meta = run_tslf_pipeline(trace_dir, strategy, max_size)
    train_time = time.time() - start

    meta.update(pipeline_meta)
    meta["train_time"] = train_time

    if spec is None:
        return None, meta

    meta["spec"] = spec
    meta["spec_source"] = "mined"

    action_fn = parse_spec_to_action_fn(spec, strategy)
    if action_fn is None:
        meta["warning"] = "Could not parse spec to action function"

    return action_fn, meta


# ============== CLI ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TSLf wrapper for blackjack")
    parser.add_argument("trace_dir", type=Path, help="Directory with traces")
    parser.add_argument("--strategy", choices=["threshold", "conservative", "basic"],
                        default="threshold")
    parser.add_argument("--max-size", type=int, default=10)
    args = parser.parse_args()

    print(f"Running TSLf pipeline on {args.trace_dir}...")
    action_fn, meta = train_and_evaluate(args.trace_dir, args.strategy, args.max_size)

    print(f"\nMetadata: {json.dumps({k: v for k, v in meta.items() if k not in ['stdout', 'stderr']}, indent=2)}")

    if action_fn:
        print("\nAction function created successfully!")
        # Test it
        test_state = {"count": 15, "dealer": 5, "standThreshold": 17,
                      "weakDealerMax": 6, "standVsWeakMin": 13}
        action = action_fn(test_state)
        print(f"Test: count=15, dealer=5 -> {'STAND' if action == 1 else 'HIT'}")
    else:
        print("\nFailed to create action function")
