#!/usr/bin/env python3
"""
Spec Transformer: Converts mined TSL specs to Issy-compatible format.

This is a standalone script that can be used to transform specs from:
1. Files (--input flag)
2. Directories containing liveness.tsl/safety.tsl/spec.tsl
3. Direct spec strings (--spec flag for quick testing)

Transformations applied:
1. Strong next to weak next: X[!] -> X
2. Safety U END pattern: (...) U (END) -> G (...)
3. Tuple equality expansion: eqC tuple1 tuple2 -> ((eq t1x t2x) && (eq t1y t2y))
4. Coordinate variable normalization: playerX -> x, playerY -> y
5. Case normalization: goalX -> goalx, hole0X -> hole0x
6. Single to double operators: & -> &&, | -> ||

Usage:
    # Transform a spec string directly (for quick testing)
    python spec_transformer.py --spec "F (eqC player goal)" --game frozen_lake

    # Transform a single file
    python spec_transformer.py path/to/spec.tsl --safety

    # Transform a directory with liveness.tsl and safety.tsl
    python spec_transformer.py path/to/out/ -o transformed.tsl

    # Verbose mode to see what transformations are applied
    python spec_transformer.py --spec "X[!] (foo)" -v
"""

import re
import argparse
import sys
from pathlib import Path


# ============== Game-Specific Configurations ==============

GAME_CONFIGS = {
    "frozen_lake": {
        # Tuple variables and their coordinate names
        "tuple_vars": {
            "player": ("x", "y"),      # player -> x, y
            "goal": ("goalx", "goaly"),
            "hole0": ("hole0x", "hole0y"),
            "hole1": ("hole1x", "hole1y"),
            "hole2": ("hole2x", "hole2y"),
            "hole3": ("hole3x", "hole3y"),
            "hole4": ("hole4x", "hole4y"),
        },
        # Variable name mappings (for already-expanded variables)
        # Includes both playerX format and player[0] format
        "var_mappings": {
            # Array-style format (from mined specs)
            "player[0]": "x",
            "player[1]": "y",
            "goal[0]": "goalx",
            "goal[1]": "goaly",
            "hole0[0]": "hole0x",
            "hole0[1]": "hole0y",
            "hole1[0]": "hole1x",
            "hole1[1]": "hole1y",
            "hole2[0]": "hole2x",
            "hole2[1]": "hole2y",
            "hole3[0]": "hole3x",
            "hole3[1]": "hole3y",
            "hole4[0]": "hole4x",
            "hole4[1]": "hole4y",
            # Legacy XY suffix format
            "playerX": "x",
            "playerY": "y",
            "goalX": "goalx",
            "goalY": "goaly",
            "hole0X": "hole0x",
            "hole0Y": "hole0y",
            "hole1X": "hole1x",
            "hole1Y": "hole1y",
            "hole2X": "hole2x",
            "hole2Y": "hole2y",
            "hole3X": "hole3x",
            "hole3Y": "hole3y",
            "hole4X": "hole4x",
            "hole4Y": "hole4y",
        },
        # Function name mappings (from mined specs to Issy format)
        "func_mappings": {
            "inc1": "add",
            "dec1": "sub",
        },
    },
    "taxi": {
        # Taxi uses coordinate constants for colored cells
        # eqC taxi locB -> (eq x BLUE_X && eq y BLUE_Y)
        "tuple_vars": {
            "taxi": ("x", "y"),        # taxi position -> x, y
            "player": ("x", "y"),      # alias for taxi
        },
        # Special location comparisons: eqC taxi locX -> coordinate constants
        # These expand to explicit coordinate comparisons
        "location_predicates": {
            "locB": ("BLUE_X", "BLUE_Y"),
            "locY": ("YELLOW_X", "YELLOW_Y"),
            "locG": ("GREEN_X", "GREEN_Y"),
            "locR": ("RED_X", "RED_Y"),
            # destination uses constants DEST_X, DEST_Y
            "destination": ("DEST_X", "DEST_Y"),
        },
        # Variable name mappings (for already-expanded variables)
        "var_mappings": {
            # Array-style format from mined specs
            "taxi[0]": "x",
            "taxi[1]": "y",
            "player[0]": "x",
            "player[1]": "y",
        },
        # Function name mappings
        "func_mappings": {
            "inc1": "add",
            "dec1": "sub",
        },
        # Boolean predicates that should remain unchanged
        "predicates": ["passengerInTaxi"],
        # Preserve case for spec_generator constants
        "preserve_case": ["DEST_X", "DEST_Y", "PASSENGER_X", "PASSENGER_Y",
                         "START_X", "START_Y", "BOUND_MIN", "BOUND_MAX",
                         "RED_X", "RED_Y", "GREEN_X", "GREEN_Y",
                         "BLUE_X", "BLUE_Y", "YELLOW_X", "YELLOW_Y"],
    },
    "cliff_walking": {
        # Cliff walking uses direct variable names
        # spec_generator uses lowercase: goalx, goaly, cliffHeight
        "tuple_vars": {},  # No tuple expansion needed
        "var_mappings": {
            # Array-style format from mined specs (if any)
            "player[0]": "x",
            "player[1]": "y",
            # Legacy XY suffix format
            "playerX": "x",
            "playerY": "y",
            # Transform to match spec_generator's naming (lowercase)
            "goalX": "goalx",
            "goalY": "goaly",
            "cliffXMin": "cliffXMin",
            "cliffXMax": "cliffXMax",
        },
        "func_mappings": {
            "inc1": "add",
            "dec1": "sub",
            # Variant movement functions for cliff_walking --var-moves
            "dbl1": "dbl",  # x -> (x * 2) + 1
            "dec2": "sub2", # y -> y - 2
        },
        # Preserve case for these variables (spec_generator uses these exact names)
        "preserve_case": ["cliffHeight", "cliffXMin", "cliffXMax"],
    },
    "blackjack": {
        # Blackjack uses scalar variables (no tuples)
        # Variables:
        #   count: player's hand count (int)
        #   stood: whether player has stood (bool)
        #   dealer: dealer's showing card (int, 1-10)
        #
        # Constants:
        #   standThreshold: 17
        #   standVsWeakMin: 12 (conservative) or 13 (basic)
        #
        # Derived predicate (expanded for paper):
        #   isWeakDealer ≡ (2 ≤ dealer ≤ 6)
        "tuple_vars": {},  # No tuple variables
        "var_mappings": {},  # Variable names are already correct
        "func_mappings": {},  # No function transformations needed
        # Derived predicates to expand to their definitions
        "predicate_expansions": {
            "isWeakDealer": "((gte dealer 2) && (lte dealer 6))",
        },
    },
    # Generic config for unknown games
    "generic": {
        "tuple_vars": {},
        "var_mappings": {},
    },
}

# Add ice_lake as an alias for frozen_lake
GAME_CONFIGS["ice_lake"] = GAME_CONFIGS["frozen_lake"]


# ============== Core Transformation Functions ==============

def transform_strong_next(spec: str, verbose: bool = False) -> str:
    """Transform strong next X[!] to weak next X."""
    original = spec
    # Handle both X[!] and X[\!] (shell-escaped version)
    result = spec.replace('X[!]', 'X').replace(r'X[\!]', 'X')
    if verbose and result != original:
        print(f"  [strong_next] X[!] -> X")
    return result


def transform_safety_u_end(spec: str, verbose: bool = False) -> str:
    """
    Transform safety U END pattern to G.
    Pattern: (...) U (END) or ... U END -> G (...)
    """
    # Pattern 1: (something) U (END)
    match = re.match(r'^\s*\((.+)\)\s*U\s*\(END\)\s*$', spec, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        result = f'G ({inner})'
        if verbose:
            print(f"  [safety_u_end] (...) U (END) -> G (...)")
        return result

    # Pattern 2: something U END (without parens on END)
    match = re.match(r'^\s*\((.+)\)\s*U\s+END\s*$', spec, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        result = f'G ({inner})'
        if verbose:
            print(f"  [safety_u_end] (...) U END -> G (...)")
        return result

    # Pattern 3: Just U END at the end
    if re.search(r'\s*U\s*\(?\s*END\s*\)?\s*$', spec):
        result = re.sub(r'\s*U\s*\(?\s*END\s*\)?\s*$', '', spec)
        result = f'G ({result.strip()})'
        if verbose:
            print(f"  [safety_u_end] removed U END, wrapped with G")
        return result

    return spec


def transform_tuple_equality(spec: str, game_config: dict, verbose: bool = False) -> str:
    """
    Transform tuple equality predicates.
    eqC tuple1 tuple2 -> ((eq t1x t2x) && (eq t1y t2y))

    Also handles special cases:
    - Location predicates (taxi): eqC taxi locB -> blue
    - Destination constants (taxi): eqC taxi destination -> (eq x DEST_X && eq y DEST_Y)

    Only applies to actual tuple variables (not already-expanded like playerX).
    """
    tuple_vars = game_config.get("tuple_vars", {})
    location_predicates = game_config.get("location_predicates", {})

    def replace_eqc(match):
        pred = match.group(1)  # 'eqC' or 'eq'
        var1 = match.group(2)
        var2 = match.group(3)

        # Check for location predicate mappings (e.g., eqC taxi locB -> blue)
        if var2 in location_predicates:
            loc_mapping = location_predicates[var2]
            if isinstance(loc_mapping, str):
                # Simple predicate name (e.g., "blue")
                if verbose:
                    print(f"  [location_pred] {pred} {var1} {var2} -> {loc_mapping}")
                return loc_mapping
            elif isinstance(loc_mapping, tuple) and len(loc_mapping) == 2:
                # Coordinate constants (e.g., ("DEST_X", "DEST_Y"))
                const_x, const_y = loc_mapping
                if var1 in tuple_vars:
                    v1x, v1y = tuple_vars[var1]
                    result = f'(eq {v1x} {const_x} && eq {v1y} {const_y})'
                    if verbose:
                        print(f"  [dest_const] {pred} {var1} {var2} -> {result}")
                    return result

        # Check if both are tuple variables
        if var1 in tuple_vars and var2 in tuple_vars:
            v1x, v1y = tuple_vars[var1]
            v2x, v2y = tuple_vars[var2]
            result = f'((eq {v1x} {v2x}) && (eq {v1y} {v2y}))'
            if verbose:
                print(f"  [tuple_eq] {pred} {var1} {var2} -> {result}")
            return result

        # If only first is tuple and second looks like a coordinate
        if var1 in tuple_vars and var2 not in tuple_vars:
            # Check if var2 ends with X or Y (might be a mixed comparison)
            # For now, leave it as is but use lowercase predicate
            pass

        # Return unchanged but ensure we use 'eq' not 'eqC'
        return f'eq {var1} {var2}'

    # Match: eqC var1 var2 (where var1 and var2 are word characters only)
    result = re.sub(r'\b(eqC?)\s+(\w+)\s+(\w+)(?!\w)', replace_eqc, spec)

    # Also transform ltC -> lt, gtC -> gt, lteC -> lte, gteC -> gte
    result = re.sub(r'\bltC\b', 'lt', result)
    result = re.sub(r'\bgtC\b', 'gt', result)
    result = re.sub(r'\blteC\b', 'lte', result)
    result = re.sub(r'\bgteC\b', 'gte', result)

    return result


def transform_variable_names(spec: str, game_config: dict, verbose: bool = False) -> str:
    """
    Transform variable names according to game config.
    E.g., playerX -> x, goalX -> goalx, player[0] -> x
    """
    var_mappings = game_config.get("var_mappings", {})
    result = spec

    # Sort by length (longest first) to avoid partial replacements
    # e.g., replace "player[0]" before "player"
    sorted_mappings = sorted(var_mappings.items(), key=lambda x: len(x[0]), reverse=True)

    for old_name, new_name in sorted_mappings:
        # Use re.escape to handle special characters like [ and ]
        pattern = re.escape(old_name)
        # Add word boundary check only if the name doesn't contain special chars
        if old_name.isalnum():
            pattern = r'\b' + pattern + r'\b'
        if re.search(pattern, result):
            result = re.sub(pattern, new_name, result)
            if verbose:
                print(f"  [var_rename] {old_name} -> {new_name}")

    return result


def transform_function_names(spec: str, game_config: dict, verbose: bool = False) -> str:
    """
    Transform function names according to game config.
    E.g., inc1 -> add, dec1 -> sub
    """
    func_mappings = game_config.get("func_mappings", {})
    result = spec

    for old_name, new_name in func_mappings.items():
        # Match function calls: inc1 followed by space and argument
        # Transform: inc1 x -> add x i1()
        pattern = r'\b' + re.escape(old_name) + r'\s+(\w+)'
        if old_name in ['inc1', 'dec1']:
            # Special handling: inc1 x -> add x i1()
            replacement = f'{new_name} \\1 i1()'
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result)
                if verbose:
                    print(f"  [func_rename] {old_name} arg -> {new_name} arg i1()")
        else:
            # Generic function renaming
            if re.search(pattern, result):
                result = re.sub(pattern, f'{new_name} \\1', result)
                if verbose:
                    print(f"  [func_rename] {old_name} -> {new_name}")

    return result


def transform_operators(spec: str, verbose: bool = False) -> str:
    """
    Transform single operators to double operators.
    & -> && (but not && -> &&&&)
    | -> || (but not || -> ||||)

    Preserves -> and <->
    """
    original = spec

    # Temporarily replace protected patterns
    result = spec.replace('&&', '\x00AND\x00')
    result = result.replace('||', '\x00OR\x00')
    result = result.replace('<->', '\x00IFF\x00')
    result = result.replace('->', '\x00IMP\x00')

    # Replace single operators
    result = result.replace('&', '&&')
    result = result.replace('|', '||')

    # Restore protected patterns
    result = result.replace('\x00AND\x00', '&&')
    result = result.replace('\x00OR\x00', '||')
    result = result.replace('\x00IFF\x00', '<->')
    result = result.replace('\x00IMP\x00', '->')

    if verbose and result != original:
        print(f"  [operators] & -> &&, | -> ||")

    return result


def expand_derived_predicates(spec: str, game_config: dict, verbose: bool = False) -> str:
    """
    Expand derived predicates to their low-level definitions.

    For example, in blackjack:
        isWeakDealer -> ((gte dealer 2) && (lte dealer 6))

    This is used for paper presentation to show the actual low-level meaning.
    """
    predicate_expansions = game_config.get("predicate_expansions", {})
    result = spec

    for predicate, expansion in predicate_expansions.items():
        # Replace the predicate with its expansion
        # Use word boundary to avoid partial matches
        pattern = r'\b' + re.escape(predicate) + r'\b'
        if re.search(pattern, result):
            result = re.sub(pattern, expansion, result)
            if verbose:
                print(f"  [expand_pred] {predicate} -> {expansion}")

    return result


def transform_case(spec: str, game_config: dict | None = None, verbose: bool = False) -> str:
    """
    Ensure consistent lowercase for coordinate suffixes.
    Converts patterns like hole0X -> hole0x (if not already handled by var_mappings).

    Respects preserve_case config to skip certain variables (e.g., cliff_walking's goalX).
    """
    preserve_case = []
    if game_config:
        preserve_case = game_config.get("preserve_case", [])

    # Pattern: word followed by X or Y at the end of the word
    # This catches any remaining uppercase suffixes
    def lowercase_suffix(match):
        full_match = match.group(0)
        # Skip if this variable should preserve case
        if full_match in preserve_case:
            return full_match
        base = match.group(1)
        suffix = match.group(2).lower()
        return base + suffix

    original = spec
    result = re.sub(r'\b(\w+)(X|Y)\b', lowercase_suffix, spec)

    if verbose and result != original:
        print(f"  [case] normalized uppercase X/Y suffixes")

    return result


# ============== Main Transform Function ==============

def transform_spec(
    spec: str,
    is_safety: bool = False,
    game: str = "frozen_lake",
    verbose: bool = False
) -> str:
    """
    Transform a mined TSL spec to Issy-compatible format.

    Args:
        spec: The mined specification string
        is_safety: Whether this is a safety spec (has U END pattern)
        game: Game name for game-specific transformations
        verbose: Print transformation steps

    Returns:
        Transformed specification string
    """
    if verbose:
        print(f"Transforming spec (game={game}, is_safety={is_safety}):")
        print(f"  Input: {spec.strip()}")

    game_config = GAME_CONFIGS.get(game, GAME_CONFIGS["generic"])
    result = spec.strip()

    # Step 1: Handle safety U END pattern (must be done first)
    if is_safety:
        result = transform_safety_u_end(result, verbose)

    # Step 2: Transform strong next X[!] to weak next X
    result = transform_strong_next(result, verbose)

    # Step 3: Transform tuple equality predicates
    result = transform_tuple_equality(result, game_config, verbose)

    # Step 4: Transform variable names
    result = transform_variable_names(result, game_config, verbose)

    # Step 5: Transform function names (inc1 -> add, dec1 -> sub)
    result = transform_function_names(result, game_config, verbose)

    # Step 6: Expand derived predicates (e.g., isWeakDealer -> gte/lte conditions)
    result = expand_derived_predicates(result, game_config, verbose)

    # Step 7: Normalize case (uppercase X/Y -> lowercase)
    result = transform_case(result, game_config, verbose)

    # Step 8: Transform single operators to double
    result = transform_operators(result, verbose)

    if verbose:
        print(f"  Output: {result}")

    return result


def transform_liveness_and_safety(
    liveness: str,
    safety: str,
    game: str = "frozen_lake",
    verbose: bool = False
) -> str:
    """
    Transform and combine liveness and safety specs.

    Args:
        liveness: The liveness spec (F-rooted)
        safety: The safety spec (... U END pattern)
        game: Game name for game-specific transformations
        verbose: Print transformation steps

    Returns:
        Combined specification: (liveness) && (safety)
    """
    trans_liveness = transform_spec(liveness.strip(), is_safety=False, game=game, verbose=verbose)
    trans_safety = transform_spec(safety.strip(), is_safety=True, game=game, verbose=verbose)

    return f'({trans_liveness}) && ({trans_safety})'


def extract_update_terms(spec: str, game: str = "frozen_lake", verbose: bool = False) -> dict:
    """
    Extract update terms from a mined spec and transform them to Issy format.

    Args:
        spec: The mined specification string
        game: Game name for game-specific transformations
        verbose: Print extraction steps

    Returns:
        Dictionary with variable names as keys and update expressions as values
        e.g., {"x": "[x <- x] || [x <- add x i1()]", "y": "[y <- y] || [y <- sub y i1()]"}
    """
    game_config = GAME_CONFIGS.get(game, GAME_CONFIGS["generic"])

    # Find all update terms in the spec: [var <- expr]
    # Parse manually to handle nested brackets like [player[0] <- inc1 player[0]]
    matches = []
    i = 0
    while i < len(spec):
        # Find the start of an update term: [var <- ...]
        if spec[i] == '[' and '<-' in spec[i:]:
            # Find the matching closing bracket
            bracket_depth = 1
            j = i + 1
            var_end = None
            arrow_pos = None

            while j < len(spec) and bracket_depth > 0:
                if spec[j] == '[':
                    bracket_depth += 1
                elif spec[j] == ']':
                    bracket_depth -= 1
                elif spec[j:j+2] == '<-' and arrow_pos is None:
                    arrow_pos = j
                    var_end = j
                j += 1

            if arrow_pos is not None and bracket_depth == 0:
                var_name = spec[i+1:var_end].strip()
                expr = spec[arrow_pos+2:j-1].strip()
                # Only include if it looks like a valid update (has a variable name)
                if var_name and not var_name.startswith('('):
                    matches.append((var_name, expr))
        i += 1

    if verbose:
        print(f"Found {len(matches)} update terms in spec")

    # Group updates by variable (after transformation)
    updates_by_var = {}

    for var_name, expr in matches:
        var_name = var_name.strip()
        expr = expr.strip()

        # Transform variable names
        transformed_var = var_name
        var_mappings = game_config.get("var_mappings", {})
        for old, new in var_mappings.items():
            if var_name == old:
                transformed_var = new
                break

        # Transform the expression
        transformed_expr = expr
        # First transform variable names in expression
        for old, new in sorted(var_mappings.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = re.escape(old)
            transformed_expr = re.sub(pattern, new, transformed_expr)

        # Then transform function names
        func_mappings = game_config.get("func_mappings", {})
        for old_func, new_func in func_mappings.items():
            if old_func in ['inc1', 'dec1']:
                pattern = r'\b' + re.escape(old_func) + r'\s+(\w+)'
                transformed_expr = re.sub(pattern, f'{new_func} \\1 i1()', transformed_expr)

        # Build the update term
        update_term = f"[{transformed_var} <- {transformed_expr}]"

        if verbose:
            print(f"  [{var_name} <- {expr}] -> {update_term}")

        # Add to collection
        if transformed_var not in updates_by_var:
            updates_by_var[transformed_var] = []
        if update_term not in updates_by_var[transformed_var]:
            updates_by_var[transformed_var].append(update_term)

    # Combine updates for each variable with ||
    result = {}
    for var, terms in updates_by_var.items():
        result[var] = " || ".join(terms)

    return result


def load_and_transform_specs(spec_dir: Path, game: str = "frozen_lake", verbose: bool = False) -> dict:
    """
    Load liveness.tsl, safety.tsl, and spec.tsl from a directory
    and return transformed versions.

    Args:
        spec_dir: Directory containing the spec files
        game: Game name for game-specific transformations
        verbose: Print transformation steps

    Returns:
        Dictionary with transformed specs
    """
    result = {}
    spec_dir = Path(spec_dir)

    liveness_file = spec_dir / 'liveness.tsl'
    safety_file = spec_dir / 'safety.tsl'
    spec_file = spec_dir / 'spec.tsl'

    if liveness_file.exists():
        liveness = liveness_file.read_text().strip()
        result['liveness'] = transform_spec(liveness, is_safety=False, game=game, verbose=verbose)
        result['liveness_original'] = liveness

    if safety_file.exists():
        safety = safety_file.read_text().strip()
        result['safety'] = transform_spec(safety, is_safety=True, game=game, verbose=verbose)
        result['safety_original'] = safety

    if spec_file.exists():
        spec = spec_file.read_text().strip()
        # Combined spec - apply general transformations
        result['combined'] = transform_spec(spec, is_safety=False, game=game, verbose=verbose)
        result['combined_original'] = spec

    # Create the final combined spec from liveness and safety
    # Only combine if both are non-empty
    has_liveness = 'liveness' in result and result['liveness'].strip()
    has_safety = 'safety' in result and result['safety'].strip()

    if has_liveness and has_safety:
        result['final'] = f"({result['liveness']}) && ({result['safety']})"
    elif has_liveness:
        result['final'] = result['liveness']
    elif has_safety:
        result['final'] = result['safety']
    elif 'combined' in result and result['combined'].strip():
        result['final'] = result['combined']
    else:
        result['final'] = None

    # Extract update terms from the original combined spec
    # These are needed for the spec_generator to know which updates to allow
    if 'combined_original' in result:
        result['variable_updates'] = extract_update_terms(result['combined_original'], game=game, verbose=verbose)
    else:
        result['variable_updates'] = {}

    return result


# ============== Unit Tests ==============

def run_tests():
    """Run unit tests for all transformations."""
    print("Running spec transformer tests...\n")

    tests_passed = 0
    tests_failed = 0

    def test(name: str, input_spec: str, expected: str, is_safety: bool = False, game: str = "frozen_lake"):
        nonlocal tests_passed, tests_failed
        result = transform_spec(input_spec, is_safety=is_safety, game=game, verbose=False)
        if result == expected:
            print(f"  PASS: {name}")
            tests_passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    Input:    {input_spec}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")
            tests_failed += 1

    # Test strong next
    print("Testing strong next transformation:")
    test("basic X[!]", "X[!] foo", "X foo")
    test("nested X[!]", "G (X[!] (X[!] bar))", "G (X (X bar))")

    # Test safety U END
    print("\nTesting safety U END transformation:")
    test("basic U END", "(foo) U (END)", "G (foo)", is_safety=True)
    test("U END no parens", "(bar) U END", "G (bar)", is_safety=True)

    # Test operator transformation
    print("\nTesting operator transformation:")
    test("single &", "a & b", "a && b")
    test("single |", "a | b", "a || b")
    test("preserve &&", "a && b", "a && b")
    test("preserve ||", "a || b", "a || b")
    test("preserve ->", "a -> b", "a -> b")
    test("preserve <->", "a <-> b", "a <-> b")
    test("mixed operators", "a & b | c -> d", "a && b || c -> d")

    # Test variable name transformation (frozen_lake)
    print("\nTesting variable name transformation (frozen_lake):")
    test("playerX", "eq playerX goalX", "eq x goalx", game="frozen_lake")
    test("playerY", "eq playerY goalY", "eq y goaly", game="frozen_lake")
    test("hole0X", "eq hole0X hole0Y", "eq hole0x hole0y", game="frozen_lake")

    # Test tuple equality (frozen_lake)
    print("\nTesting tuple equality transformation (frozen_lake):")
    test("eqC player goal", "eqC player goal", "((eq x goalx) && (eq y goaly))", game="frozen_lake")

    # Test taxi location predicates
    print("\nTesting taxi location predicates:")
    test("eqC taxi locB -> BLUE coords", "eqC taxi locB", "(eq x BLUE_X && eq y BLUE_Y)", game="taxi")
    test("eqC taxi locY -> YELLOW coords", "eqC taxi locY", "(eq x YELLOW_X && eq y YELLOW_Y)", game="taxi")
    test("eqC taxi locG -> GREEN coords", "eqC taxi locG", "(eq x GREEN_X && eq y GREEN_Y)", game="taxi")
    test("eqC taxi locR -> RED coords", "eqC taxi locR", "(eq x RED_X && eq y RED_Y)", game="taxi")
    test("eqC taxi destination -> DEST constants", "eqC taxi destination", "(eq x DEST_X && eq y DEST_Y)", game="taxi")

    # Test cliff_walking (preserves case for goalX, cliffY)
    print("\nTesting cliff_walking (case preservation):")
    test("eqC x goalX -> eq x goalX", "eqC x goalX", "eq x goalX", game="cliff_walking")
    test("eqC y cliffY -> eq y cliffY", "eqC y cliffY", "eq y cliffY", game="cliff_walking")

    # Test combined transformations
    print("\nTesting combined transformations:")
    test(
        "frozen_lake liveness",
        "F ((eqC playerX goalX) & (eqC playerY goalY))",
        "F ((eq x goalx) && (eq y goaly))",
        game="frozen_lake"
    )
    test(
        "frozen_lake liveness (tuple)",
        "F (eqC player goal)",
        "F (((eq x goalx) && (eq y goaly)))",
        game="frozen_lake"
    )
    test(
        "frozen_lake safety with U END",
        "(X[!] (! (eqC playerX hole0X))) U (END)",
        "G (X (! (eq x hole0x)))",
        is_safety=True,
        game="frozen_lake"
    )
    test(
        "frozen_lake safety with updates",
        "(X[!] (([player[1] <-  player[1]]) | (([player[0] <-  player[0]]) | (eqC player goal)))) U (END)",
        "G (X (([y <-  y]) || (([x <-  x]) || (((eq x goalx) && (eq y goaly))))))",
        is_safety=True,
        game="frozen_lake"
    )
    test(
        "taxi liveness",
        "F (eqC taxi destination)",
        "F ((eq x DEST_X && eq y DEST_Y))",
        game="taxi"
    )
    test(
        "taxi safety",
        "(X[!] ((passengerInTaxi) -> ((eqC taxi locB) <-> (eqC taxi locY)))) U (END)",
        "G (X ((passengerInTaxi) -> (((eq x BLUE_X && eq y BLUE_Y)) <-> ((eq x YELLOW_X && eq y YELLOW_Y)))))",
        is_safety=True,
        game="taxi"
    )
    test(
        "cliff_walking liveness",
        "F (G (eqC x goalX))",
        "F (G (eq x goalX))",
        game="cliff_walking"
    )
    test(
        "cliff_walking safety",
        "(X[!] ((eq y x) | ((eqC y cliffY) -> (eqC x goalX)))) U (END)",
        "G (X ((eq y x) || ((eq y cliffY) -> (eq x goalX))))",
        is_safety=True,
        game="cliff_walking"
    )

    # Summary
    print(f"\n{'='*50}")
    print(f"Tests: {tests_passed + tests_failed} total, {tests_passed} passed, {tests_failed} failed")

    return tests_failed == 0


# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(
        description='Transform mined TSL specs to Issy-compatible format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform a spec string directly (for quick testing)
  python spec_transformer.py --spec "F (eqC player goal)" --game frozen_lake

  # Transform a spec string as safety spec
  python spec_transformer.py --spec "(X[!] foo) U (END)" --safety

  # Transform a single file
  python spec_transformer.py path/to/liveness.tsl

  # Transform a directory with liveness.tsl and safety.tsl
  python spec_transformer.py path/to/out/ -o transformed.tsl

  # Run unit tests
  python spec_transformer.py --test

  # Verbose mode to see transformation steps
  python spec_transformer.py --spec "X[!] (foo & bar)" -v
"""
    )

    parser.add_argument('input', nargs='?', type=str,
                        help='Input spec file or directory containing spec files')
    parser.add_argument('--spec', '-s', type=str,
                        help='Direct spec string to transform (for quick testing)')
    parser.add_argument('--safety', action='store_true',
                        help='Treat as safety spec (has U END pattern)')
    parser.add_argument('--game', '-g', type=str, default='frozen_lake',
                        choices=list(GAME_CONFIGS.keys()),
                        help='Game for game-specific transformations (default: frozen_lake)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (default: stdout)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print transformation steps')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run unit tests')

    args = parser.parse_args()

    # Run tests if requested
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # If --spec is used and input looks like a game name, treat it as --game
    if args.spec and args.input and args.input in GAME_CONFIGS:
        if args.game == 'frozen_lake':  # Only override if game was default
            args.game = args.input
            args.input = None

    # Transform direct spec string
    if args.spec:
        transformed = transform_spec(
            args.spec,
            is_safety=args.safety,
            game=args.game,
            verbose=args.verbose
        )

        if not args.verbose:
            print(f"Input:  {args.spec}")
            print(f"Output: {transformed}")

        if args.output:
            Path(args.output).write_text(transformed)
            print(f"\nWritten to: {args.output}")

        return

    # Need either --spec or input path
    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)

    if input_path.is_dir():
        # Load from directory
        specs = load_and_transform_specs(input_path, game=args.game, verbose=args.verbose)

        print("=== Transformed Specs ===\n")

        if 'liveness_original' in specs:
            print(f"Liveness (original):    {specs['liveness_original']}")
            print(f"Liveness (transformed): {specs['liveness']}\n")

        if 'safety_original' in specs:
            print(f"Safety (original):    {specs['safety_original']}")
            print(f"Safety (transformed): {specs['safety']}\n")

        if 'final' in specs and specs['final']:
            print(f"Final combined: {specs['final']}")

            if args.output:
                Path(args.output).write_text(specs['final'])
                print(f"\nWritten to: {args.output}")
    else:
        # Load single file
        spec = input_path.read_text().strip()
        transformed = transform_spec(
            spec,
            is_safety=args.safety,
            game=args.game,
            verbose=args.verbose
        )

        if not args.verbose:
            print(f"Original:    {spec}")
            print(f"Transformed: {transformed}")

        if args.output:
            Path(args.output).write_text(transformed)
            print(f"\nWritten to: {args.output}")


if __name__ == '__main__':
    main()
