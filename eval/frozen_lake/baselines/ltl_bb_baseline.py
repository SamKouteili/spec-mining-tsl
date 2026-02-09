#!/usr/bin/env python3
"""
LTL Bit-Blasting Baseline for TSL_f Specification Mining Evaluation

This module implements a baseline that converts integer traces to bit-blasted
Boolean traces and mines LTL specifications over the individual bits.

The key insight this baseline demonstrates:
- Bit-blasting loses semantic information about the actual values
- The mined specs are unlikely to be meaningful or synthesizable
- This highlights why TSL_f's function discovery approach is important

For frozen_lake:
    Variables (integers 0-8, need 4 bits each):
        - playerX, playerY (player position)
        - goalX, goalY (goal position)
        - hole0X, hole0Y, hole1X, hole1Y, hole2X, hole2Y (hole positions)

    Total bits: 10 variables * 4 bits = 40 atomic propositions
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any


# Number of bits needed to represent values 0-8
NUM_BITS = 4
MAX_VALUE = 8


def int_to_bits(value: int, num_bits: int = NUM_BITS) -> List[int]:
    """Convert an integer to a list of bits (LSB first)."""
    if value < 0 or value > (2**num_bits - 1):
        value = min(max(value, 0), 2**num_bits - 1)  # Clamp to valid range
    bits = []
    for i in range(num_bits):
        bits.append((value >> i) & 1)
    return bits


def bits_to_int(bits: List[int]) -> int:
    """Convert a list of bits (LSB first) back to an integer."""
    value = 0
    for i, bit in enumerate(bits):
        if bit:
            value |= (1 << i)
    return value


# ============== Bit-Blasted Trace Conversion ==============

def convert_traces_to_bitblasted(trace_dir: Path, game: str = "frozen_lake") -> Dict[str, List[Dict[str, List[int]]]]:
    """
    Convert JSONL traces to bit-blasted Boolean traces.

    Each integer variable is converted to NUM_BITS individual Boolean APs.
    For example, playerX=3 becomes:
        playerX_b0=1, playerX_b1=1, playerX_b2=0, playerX_b3=0 (since 3 = 0b0011)

    Args:
        trace_dir: Directory containing pos/ and neg/ subdirectories with JSONL files
        game: Game type (only "frozen_lake" supported currently)

    Returns:
        Dict with "positive_traces" and "negative_traces" lists
    """
    if game != "frozen_lake":
        raise NotImplementedError(f"Bit-blasting baseline not yet implemented for {game}")

    result = {
        "positive_traces": [],
        "negative_traces": []
    }

    pos_dir = trace_dir / "pos"
    neg_dir = trace_dir / "neg"

    # Process positive traces
    if pos_dir.exists():
        for trace_file in sorted(pos_dir.glob("*.jsonl")):
            bitblasted_trace = _convert_single_trace_bitblasted(trace_file, game)
            if bitblasted_trace:
                result["positive_traces"].append(bitblasted_trace)

    # Process negative traces
    if neg_dir.exists():
        for trace_file in sorted(neg_dir.glob("*.jsonl")):
            bitblasted_trace = _convert_single_trace_bitblasted(trace_file, game)
            if bitblasted_trace:
                result["negative_traces"].append(bitblasted_trace)

    return result


def _convert_single_trace_bitblasted(trace_file: Path, game: str) -> Optional[Dict[str, List[int]]]:
    """
    Convert a single JSONL trace to bit-blasted Boolean representation.

    Returns a dict mapping AP names (var_bN) to lists of truth values (0 or 1) per timestep.
    """
    if game != "frozen_lake":
        raise NotImplementedError(f"Bit-blasting baseline not yet implemented for {game}")

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

    # Define variables to bit-blast
    # For tuple format: {"player": [x, y], "goal": [x, y], "hole0": [x, y], ...}
    # For scalar format: {"playerX": x, "playerY": y, ...}
    variables = [
        ("playerX", "player", 0),
        ("playerY", "player", 1),
        ("goalX", "goal", 0),
        ("goalY", "goal", 1),
        ("hole0X", "hole0", 0),
        ("hole0Y", "hole0", 1),
        ("hole1X", "hole1", 0),
        ("hole1Y", "hole1", 1),
        ("hole2X", "hole2", 0),
        ("hole2Y", "hole2", 1),
    ]

    # Initialize all bit APs with zeros
    aps = {}
    for var_name, _, _ in variables:
        for bit_idx in range(NUM_BITS):
            ap_name = f"{var_name}_b{bit_idx}"
            aps[ap_name] = [0] * trace_length

    # Also add END marker
    aps["END"] = [0] * trace_length

    def get_value(state, var_name, tuple_key, tuple_idx):
        """Extract integer value from state (handles both tuple and scalar formats)."""
        if tuple_key in state:
            # Tuple format: {"player": [x, y]}
            val = state[tuple_key]
            if isinstance(val, list):
                return val[tuple_idx]
            else:
                return val
        else:
            # Scalar format: {"playerX": x}
            return state.get(var_name, 0)

    # Process each timestep
    for t, state in enumerate(states):
        for var_name, tuple_key, tuple_idx in variables:
            value = get_value(state, var_name, tuple_key, tuple_idx)
            bits = int_to_bits(value)

            for bit_idx, bit_val in enumerate(bits):
                ap_name = f"{var_name}_b{bit_idx}"
                aps[ap_name][t] = bit_val

        # END marker at final timestep
        if t == trace_length - 1:
            aps["END"][t] = 1

    return aps


# ============== Bolt File Generation ==============

def create_bolt_file(bitblasted_traces: Dict[str, List[Dict[str, List[int]]]], output_path: Path) -> Path:
    """
    Create a Bolt-compatible JSON file from bit-blasted Boolean traces.

    Args:
        bitblasted_traces: Dict with "positive_traces" and "negative_traces"
        output_path: Path to write the Bolt JSON file

    Returns:
        Path to the created file
    """
    # Get all atomic propositions from first trace
    all_aps = set()
    for trace_type in ["positive_traces", "negative_traces"]:
        for trace in bitblasted_traces.get(trace_type, []):
            all_aps.update(trace.keys())

    # Sort APs: group by variable, then by bit index
    # This gives a sensible ordering like: playerX_b0, playerX_b1, ..., playerY_b0, ...
    def ap_sort_key(ap):
        if ap == "END":
            return ("zzz", 999)  # END goes last
        parts = ap.rsplit("_b", 1)
        if len(parts) == 2:
            var_name, bit_idx = parts
            return (var_name, int(bit_idx))
        return (ap, 0)

    ordered_aps = sorted(all_aps, key=ap_sort_key)

    # Find max trace length
    max_length = 0
    for trace_type in ["positive_traces", "negative_traces"]:
        for trace in bitblasted_traces.get(trace_type, []):
            if trace:
                length = len(next(iter(trace.values())))
                max_length = max(max_length, length)

    # Build Bolt format
    bolt_data = {
        "positive_traces": [],
        "negative_traces": [],
        "atomic_propositions": ordered_aps,
        "number_atomic_propositions": len(ordered_aps),
        "number_positive_traces": len(bitblasted_traces.get("positive_traces", [])),
        "number_negative_traces": len(bitblasted_traces.get("negative_traces", [])),
        "max_length_traces": max_length,
        "trace_type": "finite"
    }

    # Convert traces to Bolt format
    for trace_type in ["positive_traces", "negative_traces"]:
        for trace in bitblasted_traces.get(trace_type, []):
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
    max_size: int = 10,
    timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Run Bolt to mine LTL specifications from bit-blasted Boolean traces.

    Args:
        bolt_file: Path to Bolt JSON file
        max_size: Maximum formula size for enumeration (default 10 for bit-blasting)
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
        print(f"LTLBB mining timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"LTLBB mining error: {e}")
        return None


def _find_bolt_binary() -> Optional[Path]:
    """Find the Bolt binary in common locations."""
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

    # Fall back to PATH
    bolt_in_path = shutil.which("bolt")
    if bolt_in_path:
        return Path(bolt_in_path)

    return None


# ============== Spec Interpretation ==============

def interpret_bitblasted_spec(
    ltl_spec: Dict[str, Any],
    game: str = "frozen_lake"
) -> Dict[str, Any]:
    """
    Attempt to interpret a bit-blasted LTL spec back to a meaningful form.

    This is the key challenge: bit-blasted specs are unlikely to be interpretable
    because they operate on individual bits without semantic meaning.

    The interpretation attempts to:
    1. Detect patterns like "all bits of playerX equal all bits of goalX"
    2. Convert simple bit patterns back to equality predicates
    3. Flag specs that cannot be meaningfully interpreted

    Args:
        ltl_spec: Dict with "liveness" and "safety" formulas
        game: Game type

    Returns:
        Dict with:
            - "interpretable": bool - whether the spec can be converted to TSL_f
            - "liveness": str - interpreted liveness (or raw if not interpretable)
            - "safety": str - interpreted safety (or raw if not interpretable)
            - "final": str - combined spec (or empty if not interpretable)
            - "reason": str - explanation of interpretation result
    """
    if game != "frozen_lake":
        return {
            "interpretable": False,
            "liveness": ltl_spec.get("liveness", ""),
            "safety": ltl_spec.get("safety", ""),
            "final": "",
            "reason": f"Interpretation not implemented for {game}"
        }

    liveness = ltl_spec.get("liveness", "") or ""
    safety = ltl_spec.get("safety", "") or ""

    # Check if specs are too complex to interpret
    # Bit-blasted specs typically have individual bit references that don't map cleanly

    # Count bit references - if the spec mentions specific bits, it's likely not interpretable
    bit_refs = []
    for formula in [liveness, safety]:
        for var in ["playerX", "playerY", "goalX", "goalY", "hole0X", "hole0Y", "hole1X", "hole1Y", "hole2X", "hole2Y"]:
            for bit in range(NUM_BITS):
                if f"{var}_b{bit}" in formula:
                    bit_refs.append(f"{var}_b{bit}")

    if not bit_refs:
        # No bit references found - might be empty or trivial
        if not liveness and not safety:
            return {
                "interpretable": False,
                "liveness": "",
                "safety": "",
                "final": "",
                "reason": "No formulas mined"
            }
        return {
            "interpretable": False,
            "liveness": liveness,
            "safety": safety,
            "final": "",
            "reason": "No bit references in mined formulas"
        }

    # Check for interpretable patterns
    # Pattern 1: Equality check - all bits of var1 match all bits of var2
    # This would appear as: (var1_b0 <-> var2_b0) & (var1_b1 <-> var2_b1) & ...

    def check_equality_pattern(formula: str, var1: str, var2: str) -> bool:
        """Check if formula contains equality between two variables (all bits match)."""
        for bit in range(NUM_BITS):
            biconditional = f"({var1}_b{bit}) <-> ({var2}_b{bit})"
            biconditional_alt = f"{var1}_b{bit} <-> {var2}_b{bit}"
            if biconditional not in formula and biconditional_alt not in formula:
                return False
        return True

    # Try to find equality patterns
    interpreted_liveness = ""
    interpreted_safety = ""
    interpretation_notes = []

    # Check for player == goal pattern (reaching goal)
    if liveness:
        if check_equality_pattern(liveness, "playerX", "goalX") and check_equality_pattern(liveness, "playerY", "goalY"):
            interpreted_liveness = "F (((eq x goalx) && (eq y goaly)))"
            interpretation_notes.append("Detected goal-reaching pattern in liveness")
        else:
            interpretation_notes.append("Liveness uses individual bits - not interpretable as semantic predicate")

    if safety:
        # Check for player != hole patterns
        has_hole_avoidance = False
        for hole in ["hole0", "hole1", "hole2"]:
            if f"{hole}X_b" in safety or f"{hole}Y_b" in safety:
                has_hole_avoidance = True
                break

        if has_hole_avoidance:
            interpretation_notes.append("Safety mentions hole bits - potential hole avoidance, but bit-level")
        else:
            interpretation_notes.append("Safety pattern unclear at bit level")

    # Final determination
    if interpreted_liveness and not interpreted_safety:
        # Partial interpretation - liveness only
        return {
            "interpretable": True,
            "liveness": interpreted_liveness,
            "safety": "",
            "final": interpreted_liveness,
            "reason": "Partial interpretation: liveness only. " + " ".join(interpretation_notes)
        }
    elif interpreted_liveness and interpreted_safety:
        # Full interpretation
        return {
            "interpretable": True,
            "liveness": interpreted_liveness,
            "safety": interpreted_safety,
            "final": f"({interpreted_liveness}) && ({interpreted_safety})",
            "reason": "Full interpretation successful. " + " ".join(interpretation_notes)
        }
    else:
        # Not interpretable - the bit-level patterns don't map to semantic predicates
        return {
            "interpretable": False,
            "liveness": liveness,
            "safety": safety,
            "final": "",
            "reason": "Bit-blasted spec not interpretable as semantic predicates. " + " ".join(interpretation_notes),
            "raw_liveness": liveness,
            "raw_safety": safety
        }


# ============== Full Pipeline ==============

def run_ltlbb_baseline(
    trace_dir: Path,
    output_dir: Path,
    game: str = "frozen_lake",
    max_size: int = 10,
    timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Run the full LTL bit-blasting baseline pipeline.

    1. Convert traces to bit-blasted Boolean format
    2. Create Bolt file
    3. Mine LTL spec
    4. Attempt to interpret spec back to semantic form

    Args:
        trace_dir: Directory containing pos/ and neg/ trace subdirectories
        output_dir: Directory to write output files
        game: Game type
        max_size: Maximum formula size (larger for bit-blasting due to more APs)
        timeout: Mining timeout in seconds

    Returns:
        Dict with mined specs and metadata, or None on failure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert traces to bit-blasted format
    print("  [LTLBB] Converting traces to bit-blasted format...")
    bitblasted_traces = convert_traces_to_bitblasted(trace_dir, game)

    num_pos = len(bitblasted_traces.get("positive_traces", []))
    num_neg = len(bitblasted_traces.get("negative_traces", []))

    # Count number of APs
    num_aps = 0
    if num_pos > 0:
        num_aps = len(bitblasted_traces["positive_traces"][0])
    print(f"  [LTLBB] Converted {num_pos} positive, {num_neg} negative traces ({num_aps} atomic propositions)")

    # Step 2: Create Bolt file
    bolt_file = output_dir / "ltlbb_bolt.json"
    print(f"  [LTLBB] Creating Bolt file: {bolt_file}")
    create_bolt_file(bitblasted_traces, bolt_file)

    # Step 3: Mine LTL spec
    print(f"  [LTLBB] Mining LTL spec (max_size={max_size}, timeout={timeout}s)...")
    ltl_spec = mine_ltl_spec(bolt_file, max_size=max_size, timeout=timeout)

    if ltl_spec is None:
        print("  [LTLBB] Mining failed")
        return None

    # Save raw LTL spec
    ltl_spec_file = output_dir / "ltlbb_spec_raw.json"
    with open(ltl_spec_file, 'w') as f:
        json.dump(ltl_spec, f, indent=2)

    print(f"  [LTLBB] Raw liveness: {ltl_spec.get('liveness', '(none)')}")
    print(f"  [LTLBB] Raw safety: {ltl_spec.get('safety', '(none)')}")

    # Step 4: Attempt interpretation
    print("  [LTLBB] Attempting to interpret bit-blasted spec...")
    interpreted = interpret_bitblasted_spec(ltl_spec, game)

    # Save interpretation result
    interp_file = output_dir / "ltlbb_spec_interpreted.json"
    with open(interp_file, 'w') as f:
        json.dump(interpreted, f, indent=2)

    print(f"  [LTLBB] Interpretable: {interpreted.get('interpretable', False)}")
    print(f"  [LTLBB] Reason: {interpreted.get('reason', 'N/A')}")

    if interpreted.get("interpretable"):
        print(f"  [LTLBB] Interpreted liveness: {interpreted.get('liveness', '(none)')}")
        print(f"  [LTLBB] Interpreted safety: {interpreted.get('safety', '(none)')}")
        print(f"  [LTLBB] Interpreted final: {interpreted.get('final', '(none)')}")

        # Save interpreted spec files
        if interpreted.get("liveness"):
            (output_dir / "ltlbb_liveness.tsl").write_text(interpreted["liveness"])
        if interpreted.get("safety"):
            (output_dir / "ltlbb_safety.tsl").write_text(interpreted["safety"])
        if interpreted.get("final"):
            (output_dir / "ltlbb_combined.tsl").write_text(interpreted["final"])

    return {
        "bolt_file": str(bolt_file),
        "ltl_spec": ltl_spec,
        "interpreted": interpreted,
        "is_interpretable": interpreted.get("interpretable", False),
        "num_positive_traces": num_pos,
        "num_negative_traces": num_neg,
        "num_atomic_propositions": num_aps
    }


# ============== CLI ==============

def main():
    """Command-line interface for LTL bit-blasting baseline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LTL Bit-Blasting Baseline for TSL_f Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run LTLBB baseline on existing traces
    python ltl_bb_baseline.py /path/to/traces --output /path/to/output

    # With custom max formula size (larger for bit-blasting)
    python ltl_bb_baseline.py /path/to/traces --max-size 12

    # Just convert traces (no mining)
    python ltl_bb_baseline.py /path/to/traces --convert-only
"""
    )

    parser.add_argument("trace_dir", type=Path,
                        help="Directory containing pos/ and neg/ trace subdirectories")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: trace_dir/ltlbb_out)")
    parser.add_argument("--game", "-g", type=str, default="frozen_lake",
                        choices=["frozen_lake"],
                        help="Game type (default: frozen_lake)")
    parser.add_argument("--max-size", type=int, default=10,
                        help="Maximum formula size (default: 10)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Mining timeout in seconds (default: 300)")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert traces, don't mine")

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        args.output = args.trace_dir / "ltlbb_out"

    if args.convert_only:
        # Just convert traces
        print(f"Converting traces from {args.trace_dir} to bit-blasted format...")
        bitblasted_traces = convert_traces_to_bitblasted(args.trace_dir, args.game)

        args.output.mkdir(parents=True, exist_ok=True)
        bolt_file = args.output / "ltlbb_bolt.json"
        create_bolt_file(bitblasted_traces, bolt_file)

        print(f"Bolt file written to: {bolt_file}")
        print(f"  Positive traces: {len(bitblasted_traces.get('positive_traces', []))}")
        print(f"  Negative traces: {len(bitblasted_traces.get('negative_traces', []))}")
        if bitblasted_traces.get("positive_traces"):
            print(f"  Atomic propositions: {len(bitblasted_traces['positive_traces'][0])}")
    else:
        # Run full pipeline
        result = run_ltlbb_baseline(
            args.trace_dir,
            args.output,
            game=args.game,
            max_size=args.max_size,
            timeout=args.timeout
        )

        if result:
            print("\n" + "=" * 60)
            print("LTL Bit-Blasting Baseline Complete")
            print("=" * 60)
            print(f"  Output: {args.output}")
            print(f"  Atomic propositions: {result.get('num_atomic_propositions', 'N/A')}")
            print(f"  Interpretable: {result.get('is_interpretable', False)}")
            print(f"  Reason: {result.get('interpreted', {}).get('reason', 'N/A')}")
        else:
            print("\nLTL Bit-Blasting Baseline Failed")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
