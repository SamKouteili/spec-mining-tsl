#!/usr/bin/env python3
"""
Generate Popper examples for CliffWalking - v6.
Adds complement predicates (not_X) to enable learning without explicit negation.
"""

import argparse
from pathlib import Path


def compute_predicates(state):
    """Compute all relevant predicates for a state."""
    x, y = state['x'], state['y']
    goalx, goaly = state['goalX'], state['goalY']
    cxmin, cxmax = state['cliffXMin'], state['cliffXMax']
    ch = state['cliffHeight']

    preds = {}
    preds['at_goal'] = (x == goalx and y == goaly)
    preds['at_goal_x'] = (x == goalx)
    preds['not_at_goal_x'] = (x != goalx)
    preds['left_of_goal'] = (x < goalx)
    preds['above_goal_y'] = (y > goaly)
    preds['above_cliff'] = (y >= ch)
    preds['cliff_danger'] = (cxmin <= x <= cxmax and y <= ch)
    preds['safe'] = not preds['cliff_danger']  # complement of cliff_danger
    preds['at_start'] = (x == 0 and y == 0)
    preds['not_at_start'] = not preds['at_start']

    return preds


def state_id(state):
    """Create a numeric ID for a state."""
    return state['x'] * 100 + state['y'] * 10 + state['cliffHeight']


def get_optimal_action(state):
    """Get THE optimal action for a state."""
    x, y = state['x'], state['y']
    goalx, goaly = state['goalX'], state['goalY']
    cxmin, cxmax = state['cliffXMin'], state['cliffXMax']
    ch = state['cliffHeight']

    if x == goalx and y == goaly:
        return None

    # In danger zone or at start - go up
    if (cxmin <= x <= cxmax and y <= ch) or (x == 0 and y == 0):
        return 'up'

    # At goal column above goal - go down
    if x == goalx and y > goaly:
        return 'down'

    # Above cliff, left of goal, and safe - go right
    if y >= ch and x < goalx:
        return 'right'

    return None


def generate_examples(output_dir):
    """Generate Popper examples with separate action predicates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_states = {}  # sid -> (state, preds)

    # Generate states for standard board
    for x in range(12):
        for y in range(4):
            ch = 1
            state = {
                'x': x, 'y': y,
                'goalX': 11, 'goalY': 0,
                'cliffXMin': 1, 'cliffXMax': 10,
                'cliffHeight': ch
            }

            # Skip cliff positions
            if 1 <= x <= 10 and y < ch:
                continue

            sid = state_id(state)
            all_states[sid] = (state, compute_predicates(state))

    pred_names = ['at_goal', 'at_goal_x', 'not_at_goal_x', 'left_of_goal',
                  'above_goal_y', 'above_cliff', 'cliff_danger', 'safe',
                  'at_start', 'not_at_start']

    # Create separate example files for each action
    for action in ['up', 'down', 'right']:
        pos = set()
        neg = set()

        for sid, (state, preds) in all_states.items():
            opt = get_optimal_action(state)

            if opt == action:
                pos.add(sid)
            elif opt is not None:
                neg.add(sid)

        action_dir = output_dir / f'should_{action}'
        action_dir.mkdir(exist_ok=True, parents=True)

        # Write BK
        with open(action_dir / 'bk.pl', 'w') as f:
            f.write(f"%% BK for should_{action}\n\n")

            for pred in pred_names:
                f.write(f"%% {pred}\n")
                for sid, (state, preds) in sorted(all_states.items()):
                    if preds[pred]:
                        f.write(f"{pred}({sid}).\n")
                f.write("\n")

        # Write examples
        with open(action_dir / 'exs.pl', 'w') as f:
            f.write(f"%% Positive examples for should_{action}\n")
            for sid in sorted(pos):
                f.write(f"pos(should_{action}({sid})).\n")

            f.write(f"\n%% Negative examples for should_{action}\n")
            for sid in sorted(neg):
                f.write(f"neg(should_{action}({sid})).\n")

        # Write bias
        with open(action_dir / 'bias.pl', 'w') as f:
            f.write(f"""%% Bias file for should_{action}

head_pred(should_{action}, 1).

body_pred(at_goal, 1).
body_pred(at_goal_x, 1).
body_pred(not_at_goal_x, 1).
body_pred(left_of_goal, 1).
body_pred(above_goal_y, 1).
body_pred(above_cliff, 1).
body_pred(cliff_danger, 1).
body_pred(safe, 1).
body_pred(at_start, 1).
body_pred(not_at_start, 1).

max_body(3).
max_clauses(3).
max_vars(2).
""")

        print(f"  should_{action}: {len(pos)} pos, {len(neg)} neg")

    print(f"\nGenerated {len(all_states)} unique states")
    print(f"Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='work6', help='Output directory')
    args = parser.parse_args()

    generate_examples(args.output)


if __name__ == '__main__':
    main()
