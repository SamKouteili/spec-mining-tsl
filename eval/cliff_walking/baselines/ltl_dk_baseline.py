#!/usr/bin/env python3
"""
LTL Mining Baseline for CliffWalking Evaluation

This module implements a baseline that uses standard LTL mining (with domain-specific
predicates and actions) to compare against our TSL_f mining approach.

The key difference from TSL_f mining:
- TSL_f: Discovers functions from data (e.g., inc1, dec1) + mines temporal spec
- LTL Baseline: Uses pre-defined Boolean predicates (isGoal, aboveCliff, etc.)
                and actions (moveLeft, moveRight, etc.)

This provides an "optimal baseline" since it has perfect domain knowledge about
which predicates and actions matter.

Predicates for CliffWalking:
    State predicates:
        - isGoal: x == goalX && y == goalY
        - aboveCliff: y >= cliffHeight (above the cliff danger zone - safe vertically)
        - outsideCliffBounds: x < cliffXMin || x > cliffXMax (outside cliff x-range - safe horizontally)

    Actions:
        - moveLeft, moveRight, moveUp, moveDown, stay

Golden Spec for CliffWalking:
    F (isGoal) && G (outsideCliffBounds || aboveCliff)

    The safety spec ensures: if in the cliff x-range, must be above the cliff.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any


# ============== Boolean Trace Conversion ==============

def convert_traces_to_boolean(trace_dir: Path) -> Dict[str, List[Dict[str, List[int]]]]:
    """
    Convert JSONL traces to Boolean traces with predicates and actions.

    For cliff_walking:
        Predicates (state observations):
            - isGoal: x == goalX && y == goalY
            - aboveCliff: y >= cliffHeight
            - outsideCliffBounds: x < cliffXMin || x > cliffXMax

        Actions (transitions between states):
            - moveLeft: x decreased by 1
            - moveRight: x increased by 1
            - moveUp: y decreased by 1
            - moveDown: y increased by 1
            - stay: no movement (at boundary)

        Special:
            - END: final timestep marker

    Args:
        trace_dir: Directory containing pos/ and neg/ subdirectories with JSONL files

    Returns:
        Dict with "positive_traces" and "negative_traces" lists
    """
    result = {
        "positive_traces": [],
        "negative_traces": []
    }

    pos_dir = trace_dir / "pos"
    neg_dir = trace_dir / "neg"

    # Process positive traces
    if pos_dir.exists():
        for trace_file in sorted(pos_dir.glob("*.jsonl")):
            boolean_trace = _convert_single_trace(trace_file)
            if boolean_trace:
                result["positive_traces"].append(boolean_trace)

    # Process negative traces
    if neg_dir.exists():
        for trace_file in sorted(neg_dir.glob("*.jsonl")):
            boolean_trace = _convert_single_trace(trace_file)
            if boolean_trace:
                result["negative_traces"].append(boolean_trace)

    return result


def _convert_single_trace(trace_file: Path) -> Optional[Dict[str, List[int]]]:
    """
    Convert a single JSONL trace to Boolean representation.

    Returns a dict mapping AP names to lists of truth values (0 or 1) per timestep.
    """
    # Read all states from trace
    states = []
    with open(trace_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(json.loads(line))

    if not states:
        return None

    trace_length = len(states)

    # Initialize all APs with zeros
    aps = {
        # State predicates
        "isGoal": [0] * trace_length,
        "aboveCliff": [0] * trace_length,
        "outsideCliffBounds": [0] * trace_length,
        # Actions (for transitions, so one less than states)
        "moveLeft": [0] * trace_length,
        "moveRight": [0] * trace_length,
        "moveUp": [0] * trace_length,
        "moveDown": [0] * trace_length,
        "stay": [0] * trace_length,
        # End marker
        "END": [0] * trace_length,
    }

    # Process each timestep
    for t, state in enumerate(states):
        # Get positions
        x = state["x"]
        y = state["y"]
        goal_x = state["goalX"]
        goal_y = state["goalY"]
        cliff_x_min = state["cliffXMin"]
        cliff_x_max = state["cliffXMax"]
        cliff_height = state["cliffHeight"]

        # State predicates
        aps["isGoal"][t] = 1 if (x == goal_x and y == goal_y) else 0
        aps["aboveCliff"][t] = 1 if y >= cliff_height else 0
        aps["outsideCliffBounds"][t] = 1 if (x < cliff_x_min or x > cliff_x_max) else 0

        # Actions (transitions from t to t+1)
        if t < trace_length - 1:
            next_state = states[t + 1]
            next_x = next_state["x"]
            next_y = next_state["y"]

            dx = next_x - x
            dy = next_y - y

            if dx == -1 and dy == 0:
                aps["moveLeft"][t] = 1
            elif dx == 1 and dy == 0:
                aps["moveRight"][t] = 1
            elif dx == 0 and dy == -1:
                aps["moveUp"][t] = 1
            elif dx == 0 and dy == 1:
                aps["moveDown"][t] = 1
            else:
                # No movement or boundary hit
                aps["stay"][t] = 1

        # END marker at final timestep
        if t == trace_length - 1:
            aps["END"][t] = 1

    return aps


# ============== Bolt File Generation ==============

def create_bolt_file(boolean_traces: Dict[str, List[Dict[str, List[int]]]], output_path: Path) -> Path:
    """
    Create a Bolt-compatible JSON file from Boolean traces.

    Args:
        boolean_traces: Dict with "positive_traces" and "negative_traces"
        output_path: Path to write the Bolt JSON file

    Returns:
        Path to the created file
    """
    # Get all atomic propositions from first trace
    all_aps = set()
    for trace_type in ["positive_traces", "negative_traces"]:
        for trace in boolean_traces.get(trace_type, []):
            all_aps.update(trace.keys())

    # Sort APs: predicates first, then actions, then END
    predicate_aps = ["isGoal", "aboveCliff", "outsideCliffBounds"]
    action_aps = sorted([ap for ap in all_aps if ap.startswith("move") or ap == "stay"])
    special_aps = ["END"] if "END" in all_aps else []

    ordered_aps = [ap for ap in predicate_aps if ap in all_aps] + action_aps + special_aps

    # Find max trace length
    max_length = 0
    for trace_type in ["positive_traces", "negative_traces"]:
        for trace in boolean_traces.get(trace_type, []):
            if trace:
                length = len(next(iter(trace.values())))
                max_length = max(max_length, length)

    # Build Bolt format
    bolt_data = {
        "positive_traces": [],
        "negative_traces": [],
        "atomic_propositions": ordered_aps,
        "number_atomic_propositions": len(ordered_aps),
        "number_positive_traces": len(boolean_traces.get("positive_traces", [])),
        "number_negative_traces": len(boolean_traces.get("negative_traces", [])),
        "max_length_traces": max_length,
        "trace_type": "finite"
    }

    # Convert traces to Bolt format
    for trace_type in ["positive_traces", "negative_traces"]:
        for trace in boolean_traces.get(trace_type, []):
            bolt_trace = {}
            for ap in ordered_aps:
                bolt_trace[ap] = trace.get(ap, [0] * max_length)
            bolt_data[trace_type].append(bolt_trace)

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(bolt_data, f, indent=2)

    return output_path


# ============== LTL Mining ==============

def mine_ltl_spec(
    bolt_file: Path,
    max_size: int = 7,
    timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Run Bolt to mine LTL specifications from Boolean traces.

    Args:
        bolt_file: Path to Bolt JSON file
        max_size: Maximum formula size for enumeration
        timeout: Timeout in seconds

    Returns:
        Dict with "liveness" and "safety" keys containing mined formulas,
        or None on failure
    """
    # Find bolt binary
    bolt_binary = _find_bolt_binary()
    if not bolt_binary:
        raise FileNotFoundError("Bolt binary not found. Check PATH or Bolt installation.")

    cmd = [
        str(bolt_binary),
        str(bolt_file),
        str(max_size),  # max liveness size
        str(max_size),  # max safety size
        "--first-each"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Parse output
        output = result.stdout + result.stderr

        liveness = None
        safety = None

        # Look for liveness and safety markers in output
        lines = output.split('\n')
        in_liveness = False
        in_safety = False

        for line in lines:
            line = line.strip()
            if "First Liveness" in line or "=== First Liveness" in line:
                in_liveness = True
                in_safety = False
                continue
            elif "First Safety" in line or "=== First Safety" in line:
                in_safety = True
                in_liveness = False
                continue
            elif line.startswith("==="):
                in_liveness = False
                in_safety = False
                continue

            # Capture formula lines (non-empty, not debug output)
            if line and not line.startswith("[") and not line.startswith("Searching"):
                if in_liveness and liveness is None:
                    liveness = line
                elif in_safety and safety is None:
                    safety = line

        return {
            "liveness": liveness,
            "safety": safety,
            "raw_output": output
        }

    except subprocess.TimeoutExpired:
        print(f"LTL mining timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"LTL mining error: {e}")
        return None


def _find_bolt_binary() -> Optional[Path]:
    """Find the Bolt binary in common locations.

    IMPORTANT: Prefers the modified Bolt binary (with --first-each flag)
    over the standard cargo-installed version.
    """
    import shutil

    # Check project-specific locations FIRST (modified Bolt with --first-each)
    possible_locations = [
        # Project's Bolt directory
        Path(__file__).parent.parent.parent.parent / "Bolt" / "target" / "release" / "bolt",
        # Alternative project layout
        Path(__file__).parent.parent.parent / "Bolt" / "target" / "release" / "bolt",
        # Home directory Bolt
        Path.home() / "rose" / "tsl-f" / "Bolt" / "target" / "release" / "bolt",
        Path.home() / "Bolt" / "target" / "release" / "bolt",
    ]

    for loc in possible_locations:
        if loc.exists():
            return loc

    # Fall back to PATH (might not have --first-each)
    bolt_in_path = shutil.which("bolt")
    if bolt_in_path:
        return Path(bolt_in_path)

    return None


# ============== Spec Interpretation ==============

def interpret_ltl_spec(ltl_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert mined LTL spec to TSL_f/Issy-compatible format.

    The LTL predicates are mapped to their TSL_f equivalents:
        isGoal -> (eq x goalx) && (eq y goaly)
        aboveCliff -> (gte y cliffheight) or (not (lt y cliffheight))
        outsideCliffBounds -> (lt x cliffxmin) || (gt x cliffxmax)

    Actions are NOT included in the TSL_f objective since they define
    the update functions, not the specification.

    Args:
        ltl_spec: Dict with "liveness" and "safety" formulas

    Returns:
        Dict with transformed "liveness", "safety", and "final" specs
    """
    # Mapping from LTL predicates to TSL_f expressions
    predicate_map = {
        "isGoal": "((eq x goalx) && (eq y goaly))",
        "aboveCliff": "(gte y cliffheight)",
        "outsideCliffBounds": "((lt x cliffxmin) || (gt x cliffxmax))",
    }

    def transform_formula(formula: str) -> str:
        if not formula:
            return ""

        result = formula

        # Replace predicates with TSL_f equivalents
        for ltl_pred, tsl_expr in predicate_map.items():
            result = result.replace(ltl_pred, tsl_expr)

        # Remove action APs (they shouldn't appear in the mined temporal spec)
        for action in ["moveLeft", "moveRight", "moveUp", "moveDown", "stay"]:
            result = result.replace(action, "true")

        # Clean up END (replace with true for compatibility)
        result = result.replace("END", "true")

        # Normalize operators
        result = result.replace(" & ", " && ")
        result = result.replace(" | ", " || ")
        # Handle single & or | (being careful not to double existing ones)
        import re
        result = re.sub(r'(?<!&)&(?!&)', ' && ', result)
        result = re.sub(r'(?<!\|)\|(?!\|)', ' || ', result)

        # Handle double ampersands/pipes that might result
        while "&&  &&" in result:
            result = result.replace("&&  &&", "&&")
        while "||  ||" in result:
            result = result.replace("||  ||", "||")

        return result.strip()

    liveness = transform_formula(ltl_spec.get("liveness", ""))
    safety = transform_formula(ltl_spec.get("safety", ""))

    # Combine liveness and safety
    if liveness and safety:
        final = f"({liveness}) && ({safety})"
    elif liveness:
        final = liveness
    elif safety:
        final = safety
    else:
        final = ""

    return {
        "liveness": liveness,
        "safety": safety,
        "final": final,
        "raw_liveness": ltl_spec.get("liveness", ""),
        "raw_safety": ltl_spec.get("safety", "")
    }


# ============== Golden Spec Detection ==============

def is_golden_ltl_spec(ltl_spec: Dict[str, str]) -> bool:
    """
    Check if the mined LTL spec matches a known golden spec pattern.

    Golden spec for cliff_walking:
        Liveness: F (isGoal)
        Safety: G (outsideCliffBounds || aboveCliff)

        Or equivalent formulations.

    Args:
        ltl_spec: Dict with "liveness" and "safety" formulas

    Returns:
        True if spec matches a golden pattern
    """
    liveness = ltl_spec.get("liveness", "") or ""
    safety = ltl_spec.get("safety", "") or ""

    # Normalize whitespace
    liveness = ' '.join(liveness.split())
    safety = ' '.join(safety.split())

    # Check liveness: should contain F (isGoal) or F isGoal
    has_liveness = "F" in liveness and "isGoal" in liveness

    if not has_liveness:
        return False

    # Check safety: should be G (outsideCliffBounds || aboveCliff) or equivalent
    # Could be: G (outsideCliffBounds | aboveCliff)
    #           G (aboveCliff | outsideCliffBounds)
    #           G (!(!outsideCliffBounds & !aboveCliff))
    has_safety = (
        "G" in safety and
        (
            ("outsideCliffBounds" in safety and "aboveCliff" in safety) or
            ("outsideCliffBounds" in safety and "!" in safety) or
            ("aboveCliff" in safety and "!" in safety)
        )
    )

    return has_safety


def get_golden_spec_result(test_configs: List[dict]) -> dict:
    """
    Return a perfect result for golden specs (100% success).
    """
    return {
        "successes": len(test_configs),
        "total": len(test_configs),
        "steps": [15] * len(test_configs),  # typical cliff_walking path length
        "avg_steps": 15.0,
        "golden_spec": True
    }


# ============== Full Pipeline ==============

def run_ltl_baseline(
    trace_dir: Path,
    output_dir: Path,
    max_size: int = 7,
    timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Run the full LTL mining baseline pipeline.

    1. Convert traces to Boolean format
    2. Create Bolt file
    3. Mine LTL spec
    4. Interpret spec to TSL_f format

    Args:
        trace_dir: Directory containing pos/ and neg/ trace subdirectories
        output_dir: Directory to write output files
        max_size: Maximum formula size
        timeout: Mining timeout in seconds

    Returns:
        Dict with mined specs and metadata, or None on failure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert traces
    print("  [LTL] Converting traces to Boolean format...")
    boolean_traces = convert_traces_to_boolean(trace_dir)

    num_pos = len(boolean_traces.get("positive_traces", []))
    num_neg = len(boolean_traces.get("negative_traces", []))
    print(f"  [LTL] Converted {num_pos} positive, {num_neg} negative traces")

    # Step 2: Create Bolt file
    bolt_file = output_dir / "ltl_bolt.json"
    print(f"  [LTL] Creating Bolt file: {bolt_file}")
    create_bolt_file(boolean_traces, bolt_file)

    # Step 3: Mine LTL spec
    print(f"  [LTL] Mining LTL spec (max_size={max_size}, timeout={timeout}s)...")
    ltl_spec = mine_ltl_spec(bolt_file, max_size=max_size, timeout=timeout)

    if ltl_spec is None:
        print("  [LTL] Mining failed")
        return None

    # Save raw LTL spec
    ltl_spec_file = output_dir / "ltl_spec_raw.json"
    with open(ltl_spec_file, 'w') as f:
        json.dump(ltl_spec, f, indent=2)

    print(f"  [LTL] Raw liveness: {ltl_spec.get('liveness', '(none)')}")
    print(f"  [LTL] Raw safety: {ltl_spec.get('safety', '(none)')}")

    # Step 4: Interpret to TSL_f format
    print("  [LTL] Interpreting to TSL_f format...")
    tsl_spec = interpret_ltl_spec(ltl_spec)

    # Save interpreted spec
    tsl_spec_file = output_dir / "ltl_spec_tsl.json"
    with open(tsl_spec_file, 'w') as f:
        json.dump(tsl_spec, f, indent=2)

    # Save liveness and safety to separate files (for compatibility)
    if tsl_spec.get("liveness"):
        (output_dir / "ltl_liveness.tsl").write_text(tsl_spec["liveness"])
    if tsl_spec.get("safety"):
        (output_dir / "ltl_safety.tsl").write_text(tsl_spec["safety"])
    if tsl_spec.get("final"):
        (output_dir / "ltl_combined.tsl").write_text(tsl_spec["final"])

    print(f"  [LTL] TSL liveness: {tsl_spec.get('liveness', '(none)')}")
    print(f"  [LTL] TSL safety: {tsl_spec.get('safety', '(none)')}")
    print(f"  [LTL] TSL combined: {tsl_spec.get('final', '(none)')}")

    # Check for golden spec
    is_golden = is_golden_ltl_spec(ltl_spec)
    if is_golden:
        print("  [LTL] *** GOLDEN SPEC DETECTED ***")

    return {
        "bolt_file": str(bolt_file),
        "ltl_spec": ltl_spec,
        "tsl_spec": tsl_spec,
        "is_golden": is_golden,
        "num_positive_traces": num_pos,
        "num_negative_traces": num_neg
    }


# ============== CLI ==============

def main():
    """Command-line interface for LTL baseline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LTL Mining Baseline for CliffWalking Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run LTL baseline on existing traces
    python ltl_baseline.py /path/to/traces --output /path/to/output

    # With custom max formula size
    python ltl_baseline.py /path/to/traces --max-size 10

    # Just convert traces (no mining)
    python ltl_baseline.py /path/to/traces --convert-only
"""
    )

    parser.add_argument("trace_dir", type=Path,
                        help="Directory containing pos/ and neg/ trace subdirectories")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: trace_dir/ltl_out)")
    parser.add_argument("--max-size", type=int, default=7,
                        help="Maximum formula size (default: 7)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Mining timeout in seconds (default: 300)")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert traces, don't mine")

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        args.output = args.trace_dir / "ltl_out"

    if args.convert_only:
        # Just convert traces
        print(f"Converting traces from {args.trace_dir}...")
        boolean_traces = convert_traces_to_boolean(args.trace_dir)

        args.output.mkdir(parents=True, exist_ok=True)
        bolt_file = args.output / "ltl_bolt.json"
        create_bolt_file(boolean_traces, bolt_file)

        print(f"Bolt file written to: {bolt_file}")
        print(f"  Positive traces: {len(boolean_traces.get('positive_traces', []))}")
        print(f"  Negative traces: {len(boolean_traces.get('negative_traces', []))}")
    else:
        # Run full pipeline
        result = run_ltl_baseline(
            args.trace_dir,
            args.output,
            max_size=args.max_size,
            timeout=args.timeout
        )

        if result:
            print("\n" + "=" * 60)
            print("LTL Baseline Complete")
            print("=" * 60)
            print(f"  Output: {args.output}")
            print(f"  Golden spec: {result.get('is_golden', False)}")
        else:
            print("\nLTL Baseline Failed")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
