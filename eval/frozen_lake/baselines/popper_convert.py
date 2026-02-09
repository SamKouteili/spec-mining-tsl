#!/usr/bin/env python3
"""
Convert frozen_lake JSONL traces to Popper ILP format.

Supports tuple format: {"player": [x, y], "goal": [x, y], "hole0": [x, y], ...}

Popper requires three files:
- exs.pl: Positive and negative examples
- bk.pl: Background knowledge (predicates)
- bias.pl: Search space definition

Usage:
    python convert_traces_to_popper.py --traces path/to/traces --out path/to/popper_problem
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def get_player_pos(state: Dict) -> Tuple[int, int]:
    """Extract player position from state."""
    return state['player'][0], state['player'][1]


def get_goal_pos(state: Dict) -> Tuple[int, int]:
    """Extract goal position from state."""
    return state['goal'][0], state['goal'][1]


def get_hole_pos(state: Dict, hole_idx: int) -> Tuple[int, int]:
    """Extract hole position from state."""
    key = f'hole{hole_idx}'
    if key in state:
        return state[key][0], state[key][1]
    return -1, -1


def infer_action(state1: Dict, state2: Dict) -> Optional[str]:
    """Infer action from state transition."""
    px1, py1 = get_player_pos(state1)
    px2, py2 = get_player_pos(state2)

    dx = px2 - px1
    dy = py2 - py1

    if dx == 1 and dy == 0:
        return 'right'
    elif dx == -1 and dy == 0:
        return 'left'
    elif dx == 0 and dy == 1:
        return 'down'
    elif dx == 0 and dy == -1:
        return 'up'
    elif dx == 0 and dy == 0:
        return 'stay'
    return None


def state_to_prolog(state: Dict) -> str:
    """Convert state to Prolog term: state(PX,PY,GX,GY,H0X,H0Y,H1X,H1Y,H2X,H2Y)"""
    px, py = get_player_pos(state)
    gx, gy = get_goal_pos(state)
    h0x, h0y = get_hole_pos(state, 0)
    h1x, h1y = get_hole_pos(state, 1)
    h2x, h2y = get_hole_pos(state, 2)
    return f"state({px},{py},{gx},{gy},{h0x},{h0y},{h1x},{h1y},{h2x},{h2y})"


def state_to_flat_args(state: Dict) -> str:
    """Convert state to flat arguments: PX,PY,GX,GY.

    NOTE: We only include player and goal positions, not holes.
    Including holes (10 variables) makes the search space intractable for Popper.
    This means Popper cannot learn safety constraints (hole avoidance).

    This is a fundamental limitation compared to TSL_f, which can handle the full
    state representation through SyGuS + temporal mining.
    """
    px, py = get_player_pos(state)
    gx, gy = get_goal_pos(state)
    return f"{px},{py},{gx},{gy}"


def state_to_atom_id(state: Dict, idx: int) -> str:
    """Convert state to a unique atom ID.

    Returns an atom like 's0', 's1', etc. that uniquely identifies this state.
    The actual state data is stored separately as ground facts.
    """
    return f"s{idx}"


def generate_state_facts(state: Dict, state_id: str) -> str:
    """Generate ground facts for a state.

    This encodes player, goal, and hole positions as background knowledge
    rather than as head predicate arguments.
    """
    px, py = get_player_pos(state)
    gx, gy = get_goal_pos(state)
    h0x, h0y = get_hole_pos(state, 0)
    h1x, h1y = get_hole_pos(state, 1)
    h2x, h2y = get_hole_pos(state, 2)

    facts = []
    facts.append(f"player_x({state_id}, {px}).")
    facts.append(f"player_y({state_id}, {py}).")
    facts.append(f"goal_x({state_id}, {gx}).")
    facts.append(f"goal_y({state_id}, {gy}).")

    # Add holes as is_hole/3 facts
    if h0x >= 0:
        facts.append(f"is_hole({state_id}, {h0x}, {h0y}).")
    if h1x >= 0:
        facts.append(f"is_hole({state_id}, {h1x}, {h1y}).")
    if h2x >= 0:
        facts.append(f"is_hole({state_id}, {h2x}, {h2y}).")

    return '\n'.join(facts)


def load_traces(traces_dir: Path) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """Load positive and negative traces from directory."""
    pos_traces = []
    neg_traces = []

    pos_dir = traces_dir / 'pos'
    neg_dir = traces_dir / 'neg'

    if pos_dir.exists():
        for trace_file in sorted(pos_dir.glob('*.jsonl')):
            trace = []
            with open(trace_file) as f:
                for line in f:
                    if line.strip():
                        trace.append(json.loads(line))
            if trace:
                pos_traces.append(trace)

    if neg_dir.exists():
        for trace_file in sorted(neg_dir.glob('*.jsonl')):
            trace = []
            with open(trace_file) as f:
                for line in f:
                    if line.strip():
                        trace.append(json.loads(line))
            if trace:
                neg_traces.append(trace)

    return pos_traces, neg_traces


def extract_examples(pos_traces: List[List[Dict]], neg_traces: List[List[Dict]]) -> Tuple[set, set]:
    """Extract (state, action) examples from traces."""
    pos_examples = set()
    neg_examples = set()

    # From positive traces: all transitions are good examples
    for trace in pos_traces:
        for i in range(len(trace) - 1):
            action = infer_action(trace[i], trace[i+1])
            if action and action != 'stay':
                state_str = state_to_prolog(trace[i])
                pos_examples.add((state_str, action))

    # From negative traces: the action that led to a hole is bad
    for trace in neg_traces:
        if len(trace) < 2:
            continue

        for i in range(len(trace) - 1):
            next_state = trace[i + 1]
            px, py = get_player_pos(next_state)

            is_hole = False
            for h in range(3):
                hx, hy = get_hole_pos(next_state, h)
                if px == hx and py == hy:
                    is_hole = True
                    break

            if is_hole:
                action = infer_action(trace[i], trace[i+1])
                if action and action != 'stay':
                    state_str = state_to_prolog(trace[i])
                    neg_examples.add((state_str, action))
                break

    # Remove conflicts
    conflicts = pos_examples & neg_examples
    if conflicts:
        print(f"Warning: {len(conflicts)} conflicting examples, removing from negatives")
        neg_examples -= conflicts

    return pos_examples, neg_examples


def extract_examples_flat(pos_traces: List[List[Dict]], neg_traces: List[List[Dict]]) -> Tuple[set, set]:
    """Extract (px, py, gx, gy, action) examples from traces.

    Uses flat representation for better Popper compatibility.
    """
    pos_examples = set()
    neg_examples = set()

    # From positive traces: all transitions are good examples
    for trace in pos_traces:
        for i in range(len(trace) - 1):
            action = infer_action(trace[i], trace[i+1])
            if action and action != 'stay':
                args = state_to_flat_args(trace[i])
                pos_examples.add((args, action))

    # From negative traces: the action that led to a hole is bad
    for trace in neg_traces:
        if len(trace) < 2:
            continue

        for i in range(len(trace) - 1):
            next_state = trace[i + 1]
            px, py = get_player_pos(next_state)

            is_hole = False
            for h in range(3):
                hx, hy = get_hole_pos(next_state, h)
                if px == hx and py == hy:
                    is_hole = True
                    break

            if is_hole:
                action = infer_action(trace[i], trace[i+1])
                if action and action != 'stay':
                    args = state_to_flat_args(trace[i])
                    neg_examples.add((args, action))
                break

    # Remove conflicts
    conflicts = pos_examples & neg_examples
    if conflicts:
        print(f"Warning: {len(conflicts)} conflicting examples, removing from negatives")
        neg_examples -= conflicts

    return pos_examples, neg_examples


def _is_safe_action(state: Dict, action: str) -> bool:
    """Check if action matches the 'safe' policy (goal direction AND not blocked)."""
    px, py = get_player_pos(state)
    gx, gy = get_goal_pos(state)

    holes = []
    for i in range(3):
        hx, hy = get_hole_pos(state, i)
        if hx >= 0:
            holes.append((hx, hy))

    if action == 'right':
        return (px < gx) and (px + 1, py) not in holes
    elif action == 'left':
        return (px > gx) and (px - 1, py) not in holes
    elif action == 'down':
        return (py < gy) and (px, py + 1) not in holes
    elif action == 'up':
        return (py > gy) and (px, py - 1) not in holes
    return False


def extract_examples_stateful(pos_traces: List[List[Dict]], neg_traces: List[List[Dict]],
                               filter_to_safe: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """Extract examples using state atoms with ground facts.

    Instead of passing state info as head arguments, each state gets a unique
    atom ID (s0, s1, s2...) and all state info is stored as ground facts.

    This reduces head arity to 1, making search much more tractable.

    Args:
        pos_traces: List of positive (winning) traces
        neg_traces: List of negative (losing) traces
        filter_to_safe: If True, only include positive examples where the action
                       matches the ideal safe_X policy (goal direction AND not blocked).
                       This filters out "detour" moves that confuse ILP.

    Returns:
        (pos_examples, neg_examples) where each is a list of dicts:
        {'state_id': 's0', 'state': {...}, 'action': 'right'}
    """
    pos_examples = []
    neg_examples = []
    state_counter = [0]  # Use list for closure
    filtered_count = 0

    def get_next_id():
        sid = f"s{state_counter[0]}"
        state_counter[0] += 1
        return sid

    # From positive traces: transitions where action matches safe policy
    for trace in pos_traces:
        for i in range(len(trace) - 1):
            action = infer_action(trace[i], trace[i+1])
            if action and action != 'stay':
                # Filter: only include if action matches safe_X policy
                if filter_to_safe and not _is_safe_action(trace[i], action):
                    filtered_count += 1
                    continue

                state_id = get_next_id()
                pos_examples.append({
                    'state_id': state_id,
                    'state': trace[i],
                    'action': action
                })

    if filter_to_safe and filtered_count > 0:
        print(f"  Filtered out {filtered_count} 'detour' examples that don't match safe_X policy")

    # From negative traces: the action that led to a hole is bad
    for trace in neg_traces:
        if len(trace) < 2:
            continue

        for i in range(len(trace) - 1):
            next_state = trace[i + 1]
            px, py = get_player_pos(next_state)

            is_hole = False
            for h in range(3):
                hx, hy = get_hole_pos(next_state, h)
                if px == hx and py == hy:
                    is_hole = True
                    break

            if is_hole:
                action = infer_action(trace[i], trace[i+1])
                if action and action != 'stay':
                    state_id = get_next_id()
                    neg_examples.append({
                        'state_id': state_id,
                        'state': trace[i],
                        'action': action
                    })
                break

    return pos_examples, neg_examples


def generate_exs_pl_stateful(pos_examples: List[Dict], neg_examples: List[Dict], action: str) -> str:
    """Generate examples file using state atoms.

    Examples are: go_right(s0), go_right(s1), etc.
    State info is in background knowledge.
    """
    lines = [f"% Examples for go_{action}"]
    lines.append(f"% Format: go_{action}(State) where State is an atom")
    lines.append("")

    for ex in pos_examples:
        if ex['action'] == action:
            lines.append(f"pos(go_{action}({ex['state_id']})).")

    lines.append("")
    for ex in neg_examples:
        if ex['action'] == action:
            lines.append(f"neg(go_{action}({ex['state_id']})).")

    return '\n'.join(lines)


def generate_bk_pl_stateful(pos_examples: List[Dict], neg_examples: List[Dict],
                            max_coord: int = 4) -> str:
    """Generate background knowledge with state facts.

    Each state atom has associated ground facts:
    - player_x(s0, 1).
    - player_y(s0, 2).
    - goal_x(s0, 3).
    - goal_y(s0, 3).
    - is_hole(s0, 1, 1).
    etc.
    """
    bk = '''% Background knowledge with state atoms
% State information stored as ground facts

:- discontiguous player_x/2.
:- discontiguous player_y/2.
:- discontiguous goal_x/2.
:- discontiguous goal_y/2.
:- discontiguous is_hole/3.
:- discontiguous safe_right/1.
:- discontiguous safe_left/1.
:- discontiguous safe_down/1.
:- discontiguous safe_up/1.
:- discontiguous not_blocked_right/1.
:- discontiguous not_blocked_left/1.
:- discontiguous not_blocked_down/1.
:- discontiguous not_blocked_up/1.

'''
    # Generate state facts for all examples
    all_examples = pos_examples + neg_examples
    for ex in all_examples:
        state_id = ex['state_id']
        state = ex['state']
        bk += generate_state_facts(state, state_id) + "\n"

    # Add comparison predicates
    bk += f"\n% Comparison predicates for coordinates 0 to {max_coord}\n"
    bk += "% my_lt(X,Y) means X < Y\n"
    for x in range(max_coord + 1):
        for y in range(max_coord + 1):
            if x < y:
                bk += f"my_lt({x},{y}).\n"

    bk += "\n% my_gt(X,Y) means X > Y\n"
    for x in range(max_coord + 1):
        for y in range(max_coord + 1):
            if x > y:
                bk += f"my_gt({x},{y}).\n"

    bk += "\n% my_eq(X,Y) means X == Y\n"
    for x in range(max_coord + 1):
        bk += f"my_eq({x},{x}).\n"

    # Add increment/decrement facts
    bk += "\n% inc(X,Y) means Y = X + 1\n"
    for x in range(max_coord):
        bk += f"inc({x},{x+1}).\n"

    bk += "\n% dec(X,Y) means Y = X - 1\n"
    for x in range(1, max_coord + 1):
        bk += f"dec({x},{x-1}).\n"

    # Add derived predicates for safety
    bk += '''
% Derived: check if moving in direction from state S would hit a hole
would_hit_right(S) :- player_x(S, PX), player_y(S, PY), inc(PX, NX), is_hole(S, NX, PY).
would_hit_left(S) :- player_x(S, PX), player_y(S, PY), dec(PX, NX), is_hole(S, NX, PY).
would_hit_down(S) :- player_x(S, PX), player_y(S, PY), inc(PY, NY), is_hole(S, PX, NY).
would_hit_up(S) :- player_x(S, PX), player_y(S, PY), dec(PY, NY), is_hole(S, PX, NY).

% Derived: goal direction relative to player
goal_right(S) :- player_x(S, PX), goal_x(S, GX), my_lt(PX, GX).
goal_left(S) :- player_x(S, PX), goal_x(S, GX), my_gt(PX, GX).
goal_down(S) :- player_y(S, PY), goal_y(S, GY), my_lt(PY, GY).
goal_up(S) :- player_y(S, PY), goal_y(S, GY), my_gt(PY, GY).
goal_same_x(S) :- player_x(S, X), goal_x(S, X).
goal_same_y(S) :- player_y(S, Y), goal_y(S, Y).
'''

    # Now add SAFE direction predicates as ground facts
    # safe_right(S) = goal_right(S) AND NOT would_hit_right(S)
    # Since Popper doesn't support negation, we compute this explicitly
    bk += "\n% SAFE direction predicates (goal direction AND not blocked by hole)\n"
    bk += "% These are the KEY predicates for learning correct navigation\n"

    for ex in all_examples:
        state_id = ex['state_id']
        state = ex['state']
        px, py = get_player_pos(state)
        gx, gy = get_goal_pos(state)

        # Get holes
        holes = []
        for i in range(3):
            hx, hy = get_hole_pos(state, i)
            if hx >= 0:
                holes.append((hx, hy))

        # Check each direction
        # safe_right: goal is right AND moving right won't hit hole
        goal_right = px < gx
        would_hit_right = (px + 1, py) in holes
        if goal_right and not would_hit_right:
            bk += f"safe_right({state_id}).\n"

        # safe_left: goal is left AND moving left won't hit hole
        goal_left = px > gx
        would_hit_left = (px - 1, py) in holes
        if goal_left and not would_hit_left:
            bk += f"safe_left({state_id}).\n"

        # safe_down: goal is down AND moving down won't hit hole
        goal_down = py < gy
        would_hit_down = (px, py + 1) in holes
        if goal_down and not would_hit_down:
            bk += f"safe_down({state_id}).\n"

        # safe_up: goal is up AND moving up won't hit hole
        goal_up = py > gy
        would_hit_up = (px, py - 1) in holes
        if goal_up and not would_hit_up:
            bk += f"safe_up({state_id}).\n"

        # Also add "not blocked" predicates (for alternative rules)
        if not would_hit_right:
            bk += f"not_blocked_right({state_id}).\n"
        if not would_hit_left:
            bk += f"not_blocked_left({state_id}).\n"
        if not would_hit_down:
            bk += f"not_blocked_down({state_id}).\n"
        if not would_hit_up:
            bk += f"not_blocked_up({state_id}).\n"

    return bk


def generate_bias_pl_stateful(action: str, minimal: bool = False) -> str:
    """Generate bias for stateful representation.

    Head: go_action(State) - just 1 argument!
    Body predicates extract info from state via ground facts.

    Args:
        action: The action name
        minimal: If True, only include safe_X predicates (for cleaner learning)
    """
    if minimal:
        # MINIMAL BIAS: Only safe_X predicates - forces clean rules
        # Separate file per action with ONLY the relevant safe_X predicate
        safe_pred = f"safe_{action}"
        bias = f'''% MINIMAL Bias for go_{action} - ONLY safe_{action} predicate
% This forces Popper to learn: go_{action}(S) :- safe_{action}(S).

head_pred(go_{action}, 1).

% ONLY the matching safe predicate for this action
body_pred({safe_pred}, 1).

% Types
type(go_{action}, (state,)).
type({safe_pred}, (state,)).

% Directions
direction(go_{action}, (in,)).
direction({safe_pred}, (in,)).

% Minimal search constraints
max_clauses(1).
max_body(1).
max_vars(1).

allow_singletons.
'''
        return bias

    # FULL BIAS: All predicates
    bias = f'''% Bias for go_{action} with state atoms
% Format: go_{action}(State) where State is an atom

head_pred(go_{action}, 1).

% State property extraction
body_pred(player_x, 2).
body_pred(player_y, 2).
body_pred(goal_x, 2).
body_pred(goal_y, 2).
body_pred(is_hole, 3).

% Comparison predicates
body_pred(my_lt, 2).
body_pred(my_gt, 2).
body_pred(my_eq, 2).
body_pred(inc, 2).
body_pred(dec, 2).

% Derived predicates for safety (precomputed in BK)
body_pred(would_hit_right, 1).
body_pred(would_hit_left, 1).
body_pred(would_hit_down, 1).
body_pred(would_hit_up, 1).

% Derived predicates for goal direction
body_pred(goal_right, 1).
body_pred(goal_left, 1).
body_pred(goal_down, 1).
body_pred(goal_up, 1).
body_pred(goal_same_x, 1).
body_pred(goal_same_y, 1).

% SAFE direction predicates (goal direction AND not blocked) - KEY FOR LEARNING!
body_pred(safe_right, 1).
body_pred(safe_left, 1).
body_pred(safe_down, 1).
body_pred(safe_up, 1).

% Not blocked predicates (for alternative strategies)
body_pred(not_blocked_right, 1).
body_pred(not_blocked_left, 1).
body_pred(not_blocked_down, 1).
body_pred(not_blocked_up, 1).

% Types
type(go_{action}, (state,)).
type(player_x, (state, coord)).
type(player_y, (state, coord)).
type(goal_x, (state, coord)).
type(goal_y, (state, coord)).
type(is_hole, (state, coord, coord)).
type(my_lt, (coord, coord)).
type(my_gt, (coord, coord)).
type(my_eq, (coord, coord)).
type(inc, (coord, coord)).
type(dec, (coord, coord)).
type(would_hit_right, (state,)).
type(would_hit_left, (state,)).
type(would_hit_down, (state,)).
type(would_hit_up, (state,)).
type(goal_right, (state,)).
type(goal_left, (state,)).
type(goal_down, (state,)).
type(goal_up, (state,)).
type(goal_same_x, (state,)).
type(goal_same_y, (state,)).
type(safe_right, (state,)).
type(safe_left, (state,)).
type(safe_down, (state,)).
type(safe_up, (state,)).
type(not_blocked_right, (state,)).
type(not_blocked_left, (state,)).
type(not_blocked_down, (state,)).
type(not_blocked_up, (state,)).

% Directions
direction(go_{action}, (in,)).
direction(player_x, (in, out)).
direction(player_y, (in, out)).
direction(goal_x, (in, out)).
direction(goal_y, (in, out)).
direction(is_hole, (in, in, in)).
direction(my_lt, (in, in)).
direction(my_gt, (in, in)).
direction(my_eq, (in, in)).
direction(inc, (in, out)).
direction(dec, (in, out)).
direction(would_hit_right, (in,)).
direction(would_hit_left, (in,)).
direction(would_hit_down, (in,)).
direction(would_hit_up, (in,)).
direction(goal_right, (in,)).
direction(goal_left, (in,)).
direction(goal_down, (in,)).
direction(goal_up, (in,)).
direction(goal_same_x, (in,)).
direction(goal_same_y, (in,)).
direction(safe_right, (in,)).
direction(safe_left, (in,)).
direction(safe_down, (in,)).
direction(safe_up, (in,)).
direction(not_blocked_right, (in,)).
direction(not_blocked_left, (in,)).
direction(not_blocked_down, (in,)).
direction(not_blocked_up, (in,)).

% Search constraints - very tractable with 1 head var!
max_clauses(3).
max_body(4).
max_vars(5).

allow_singletons.
'''
    return bias


def generate_exs_pl_for_action(pos_examples: set, neg_examples: set, action: str) -> str:
    """Generate examples file for a single action."""
    lines = [f"% Examples for go_{action}"]

    for state, a in sorted(pos_examples):
        if a == action:
            lines.append(f"pos(go_{action}({state})).")

    lines.append("")
    for state, a in sorted(neg_examples):
        if a == action:
            lines.append(f"neg(go_{action}({state})).")

    return '\n'.join(lines)


def generate_exs_pl_flat(pos_examples: set, neg_examples: set, action: str) -> str:
    """Generate examples file for a single action using flat representation.

    Examples are: go_right(PX, PY, GX, GY) where args are player/goal coords.
    """
    lines = [f"% Examples for go_{action}"]
    lines.append(f"% Format: go_{action}(player_x, player_y, goal_x, goal_y)")
    lines.append("")

    for args, a in sorted(pos_examples):
        if a == action:
            lines.append(f"pos(go_{action}({args})).")

    lines.append("")
    for args, a in sorted(neg_examples):
        if a == action:
            lines.append(f"neg(go_{action}({args})).")

    return '\n'.join(lines)




def generate_bk_pl_fair(max_coord: int = 4) -> str:
    """Generate FAIR background knowledge - equivalent to what TSL_f receives.

    TSL_f gets:
    - Position extraction (raw numeric values)
    - Grid bounds (0 to max_coord)
    - Comparison operators
    - Constraint: only one axis moves at a time (implicit in per-action learning)

    TSL_f does NOT get:
    - Movement grammar (+1/-1) - it discovers this via SyGuS

    Popper must discover the movement relationships.

    NOTE: We use ground facts for comparisons instead of arithmetic rules
    because Popper can't evaluate arithmetic during recall computation.
    """
    bk = f'''% FAIR background knowledge - equivalent to TSL_f
% State: state(PX, PY, GX, GY, H0X, H0Y, H1X, H1Y, H2X, H2Y)

% Extract player position
player_x(state(PX,_,_,_,_,_,_,_,_,_), PX).
player_y(state(_,PY,_,_,_,_,_,_,_,_), PY).

% Extract goal position
goal_x(state(_,_,GX,_,_,_,_,_,_,_), GX).
goal_y(state(_,_,_,GY,_,_,_,_,_,_), GY).

% Extract hole positions
hole0_x(state(_,_,_,_,H0X,_,_,_,_,_), H0X).
hole0_y(state(_,_,_,_,_,H0Y,_,_,_,_), H0Y).
hole1_x(state(_,_,_,_,_,_,H1X,_,_,_), H1X).
hole1_y(state(_,_,_,_,_,_,_,H1Y,_,_), H1Y).
hole2_x(state(_,_,_,_,_,_,_,_,H2X,_), H2X).
hole2_y(state(_,_,_,_,_,_,_,_,_,H2Y), H2Y).

% Grid bounds - TSL_f knows coordinates are in [0, {max_coord}]
% Valid coordinate values
'''
    # Add valid coordinate facts
    for i in range(max_coord + 1):
        bk += f"valid_coord({i}).\n"

    bk += "\n% Bound checking as ground facts\n"
    bk += "at_min_bound(0).\n"
    bk += f"at_max_bound({max_coord}).\n"
    for i in range(max_coord + 1):
        bk += f"in_bounds({i}).\n"

    # Generate ground facts for comparisons (avoid arithmetic issues)
    bk += "\n% Comparison predicates as ground facts\n"
    bk += "% my_lt(X,Y) means X < Y\n"
    for x in range(max_coord + 1):
        for y in range(max_coord + 1):
            if x < y:
                bk += f"my_lt({x},{y}).\n"

    bk += "\n% my_gt(X,Y) means X > Y\n"
    for x in range(max_coord + 1):
        for y in range(max_coord + 1):
            if x > y:
                bk += f"my_gt({x},{y}).\n"

    bk += "\n% my_eq(X,Y) means X == Y\n"
    for x in range(max_coord + 1):
        bk += f"my_eq({x},{x}).\n"

    bk += f'''
% Note: Popper must DISCOVER that moves are +1/-1
% We do NOT provide successor/predecessor - that's what TSL_f discovers via SyGuS
'''
    return bk


def generate_bk_pl_flat(max_coord: int = 4, with_functions: bool = False) -> str:
    """Generate background knowledge for flat representation.

    Format: go_action(PX, PY, GX, GY)

    NOTE: Only 4 arguments (player + goal positions). Including holes (10 args)
    makes the search space intractable for Popper (~200s+ per action vs ~1s).

    Args:
        max_coord: Maximum coordinate value (grid_size - 1)
        with_functions: If True, include movement functions (+1/-1).
    """
    bk = f'''% Background knowledge for flat representation
% go_action(PX, PY, GX, GY)
% NOTE: Holes not included - 10-arg format is intractable for Popper

% Comparison predicates as ground facts (for coords 0 to {max_coord})
'''
    # Generate ground facts for comparisons
    bk += "% my_lt(X,Y) means X < Y\n"
    for x in range(max_coord + 1):
        for y in range(max_coord + 1):
            if x < y:
                bk += f"my_lt({x},{y}).\n"

    bk += "\n% my_gt(X,Y) means X > Y\n"
    for x in range(max_coord + 1):
        for y in range(max_coord + 1):
            if x > y:
                bk += f"my_gt({x},{y}).\n"

    bk += "\n% my_eq(X,Y) means X == Y\n"
    for x in range(max_coord + 1):
        bk += f"my_eq({x},{x}).\n"

    bk += f"\n% Bounds\n"
    bk += "at_min(0).\n"
    bk += f"at_max({max_coord}).\n"
    for i in range(max_coord + 1):
        bk += f"in_bounds({i}).\n"

    # Add movement functions if requested
    if with_functions:
        bk += f"\n% Movement functions (equivalent to what TSL_f discovers via SyGuS)\n"
        bk += "% inc(X, Y) means Y = X + 1\n"
        for x in range(max_coord):
            bk += f"inc({x},{x+1}).\n"

        bk += "\n% dec(X, Y) means Y = X - 1\n"
        for x in range(1, max_coord + 1):
            bk += f"dec({x},{x-1}).\n"

        bk += "\n% id(X, X) means identity\n"
        for x in range(-1, max_coord + 1):
            bk += f"id({x},{x}).\n"

    return bk


def generate_bias_pl_flat(action: str, with_functions: bool = False) -> str:
    """Generate bias for flat representation with holes.

    Args:
        action: The action name (right, left, up, down)
        with_functions: If True, include movement functions in the search space.

    Format: go_action(PX, PY, GX, GY, H0X, H0Y, H1X, H1Y, H2X, H2Y)
    Variables: V0=PX, V1=PY, V2=GX, V3=GY, V4=H0X, V5=H0Y, V6=H1X, V7=H1Y, V8=H2X, V9=H2Y
    """
    bias = f'''% Bias for go_{action} with flat representation (includes holes)
% Format: go_{action}(PX, PY, GX, GY, H0X, H0Y, H1X, H1Y, H2X, H2Y)
% V0=PX, V1=PY, V2=GX, V3=GY, V4=H0X, V5=H0Y, V6=H1X, V7=H1Y, V8=H2X, V9=H2Y

head_pred(go_{action}, 10).

% Comparison predicates
body_pred(my_lt, 2).
body_pred(my_gt, 2).
body_pred(my_eq, 2).
body_pred(at_min, 1).
body_pred(at_max, 1).
body_pred(in_bounds, 1).
'''

    if with_functions:
        bias += '''
% Movement functions (given as domain knowledge)
body_pred(inc, 2).
body_pred(dec, 2).
body_pred(id, 2).
'''

    bias += f'''
% Types - all coordinates are integers
type(go_{action}, (coord, coord, coord, coord, coord, coord, coord, coord, coord, coord)).
type(my_lt, (coord, coord)).
type(my_gt, (coord, coord)).
type(my_eq, (coord, coord)).
type(at_min, (coord,)).
type(at_max, (coord,)).
type(in_bounds, (coord,)).
'''

    if with_functions:
        bias += '''type(inc, (coord, coord)).
type(dec, (coord, coord)).
type(id, (coord, coord)).
'''

    bias += f'''
% Directions - head args are inputs, comparison args come from head
direction(go_{action}, (in, in, in, in, in, in, in, in, in, in)).
direction(my_lt, (in, in)).
direction(my_gt, (in, in)).
direction(my_eq, (in, in)).
direction(at_min, (in,)).
direction(at_max, (in,)).
direction(in_bounds, (in,)).
'''

    if with_functions:
        bias += '''direction(inc, (in, out)).
direction(dec, (in, out)).
direction(id, (in, out)).
'''

    bias += '''
% Search constraints - minimal for 10-variable tractability
max_clauses(1).
max_body(2).
max_vars(10).

allow_singletons.
'''
    return bias


def generate_bias_pl_fair(action: str) -> str:
    """Generate FAIR bias - equivalent to TSL_f's search space.

    Popper must discover movement relationships like:
    go_right(S) :- player_x(S,PX), goal_x(S,GX), my_lt(PX,GX).
    """
    return f'''% FAIR bias for go_{action}
% Popper must discover movement patterns (TSL_f discovers +1/-1 via SyGuS)

head_pred(go_{action}, 1).

% Position extraction predicates
body_pred(player_x, 2).
body_pred(player_y, 2).
body_pred(goal_x, 2).
body_pred(goal_y, 2).
body_pred(hole0_x, 2).
body_pred(hole0_y, 2).
body_pred(hole1_x, 2).
body_pred(hole1_y, 2).
body_pred(hole2_x, 2).
body_pred(hole2_y, 2).

% Comparison predicates (ground facts)
body_pred(my_lt, 2).
body_pred(my_gt, 2).
body_pred(my_eq, 2).

% Bounds predicates (TSL_f knows bounds)
body_pred(at_min_bound, 1).
body_pred(at_max_bound, 1).
body_pred(in_bounds, 1).

% Types
type(go_{action}, (state,)).
type(player_x, (state, int)).
type(player_y, (state, int)).
type(goal_x, (state, int)).
type(goal_y, (state, int)).
type(hole0_x, (state, int)).
type(hole0_y, (state, int)).
type(hole1_x, (state, int)).
type(hole1_y, (state, int)).
type(hole2_x, (state, int)).
type(hole2_y, (state, int)).
type(my_lt, (int, int)).
type(my_gt, (int, int)).
type(my_eq, (int, int)).
type(at_min_bound, (int,)).
type(at_max_bound, (int,)).
type(in_bounds, (int,)).

% Directions
direction(go_{action}, (in,)).
direction(player_x, (in, out)).
direction(player_y, (in, out)).
direction(goal_x, (in, out)).
direction(goal_y, (in, out)).
direction(hole0_x, (in, out)).
direction(hole0_y, (in, out)).
direction(hole1_x, (in, out)).
direction(hole1_y, (in, out)).
direction(hole2_x, (in, out)).
direction(hole2_y, (in, out)).
direction(my_lt, (in, in)).
direction(my_gt, (in, in)).
direction(my_eq, (in, in)).
direction(at_min_bound, (in,)).
direction(at_max_bound, (in,)).
direction(in_bounds, (in,)).

% Search constraints
max_clauses(4).
max_body(5).
max_vars(4).

allow_singletons.
'''


def main():
    parser = argparse.ArgumentParser(description='Convert frozen_lake traces to Popper format')
    parser.add_argument('--traces', type=str, required=True,
                        help='Path to traces directory (containing pos/ and neg/ subdirs)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for Popper problem files')
    args = parser.parse_args()

    traces_dir = Path(args.traces)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading traces from {traces_dir}")
    pos_traces, neg_traces = load_traces(traces_dir)
    print(f"  Loaded {len(pos_traces)} positive traces, {len(neg_traces)} negative traces")

    print("Extracting examples...")
    pos_examples, neg_examples = extract_examples(pos_traces, neg_traces)
    print(f"  {len(pos_examples)} positive examples, {len(neg_examples)} negative examples")

    # Write files for each action
    for action in ['right', 'left', 'up', 'down']:
        action_dir = out_dir / f'action_{action}'
        action_dir.mkdir(parents=True, exist_ok=True)

        with open(action_dir / 'exs.pl', 'w') as f:
            f.write(generate_exs_pl_for_action(pos_examples, neg_examples, action))

        with open(action_dir / 'bk.pl', 'w') as f:
            f.write(generate_bk_pl_fair())

        with open(action_dir / 'bias.pl', 'w') as f:
            f.write(generate_bias_pl_fair(action))

    print(f"\nPopper problem files written to {out_dir}")
    print("Run with: python baselines/popper/run_popper_baseline.py --traces <path>")


if __name__ == '__main__':
    main()
