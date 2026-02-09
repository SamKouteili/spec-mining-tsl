# Popper ILP Baseline for CliffWalking

This directory contains an Inductive Logic Programming (ILP) baseline using [Popper](https://github.com/logic-and-learning-lab/Popper) to learn a winning policy for the CliffWalking game.

## Results

**Win Rate: 100%** (100/100 test episodes)

## Learned Policy

Popper learns interpretable logical rules that form a complete winning policy:

```prolog
% Go up if in cliff danger zone OR at start position
should_up(State) :- cliff_danger(State).
should_up(State) :- at_start(State).

% Go down if at the goal's x-coordinate
should_down(State) :- at_goal_x(State).

% Go right if safe, not at start, and left of goal
should_right(State) :- safe(State), not_at_start(State), left_of_goal(State).
```

### Policy Interpretation

1. **Start**: At position (0,0), go UP to escape the cliff danger zone
2. **Cliff Edge**: While in the danger zone (x ∈ [1,10], y ≤ cliffHeight), go UP
3. **Traverse**: Once safely above the cliff, go RIGHT towards the goal
4. **Descend**: When at the goal's x-coordinate (x=11), go DOWN to reach the goal

## Background Predicates

The following predicates are provided as background knowledge:

| Predicate | Description |
|-----------|-------------|
| `at_goal(S)` | Player is at the goal position |
| `at_goal_x(S)` | Player's x equals goal's x |
| `left_of_goal(S)` | Player's x < goal's x |
| `above_goal_y(S)` | Player's y > goal's y |
| `above_cliff(S)` | Player's y >= cliff height |
| `cliff_danger(S)` | In danger zone (x in cliff range AND y <= cliff height) |
| `safe(S)` | NOT in danger zone |
| `at_start(S)` | Player is at (0,0) |

## Usage

```bash
# Run the complete baseline (generates examples, learns rules, evaluates)
python run_baseline.py --gen 10

# Use existing traces
python run_baseline.py --traces path/to/traces

# Just evaluate previously learned rules
python run_baseline.py --eval --num-tests 100

# With random cliff heights (generalization test)
python run_baseline.py --gen 20 --random-height
```

## Comparison with TSL_f Mining

| Aspect | Popper ILP | TSL_f Mining |
|--------|------------|--------------|
| **Input** | State-action pairs | State traces |
| **Output** | Prolog rules | TSL temporal spec + functions |
| **Discovery** | Logical rules from predicates | Functions + temporal formulas |
| **Predicates** | Must be provided | Discovered from transitions |
| **Interpretability** | High (logical rules) | High (temporal logic) |
| **Generalization** | Via predicate abstraction | Via function abstraction |

### Key Differences

1. **Predicate Engineering**: Popper requires predefined predicates (e.g., `cliff_danger`, `safe`) that capture domain knowledge. TSL_f mining discovers functions from data.

2. **Temporal Reasoning**: TSL_f can express temporal properties (e.g., "eventually reach goal", "always avoid holes"), while Popper learns state-action mappings.

3. **Sample Complexity**: Both methods learn from demonstrations, but Popper may need fewer examples when good predicates are provided.

## Files

- `run_baseline.py` - Main script to run the complete baseline
- `generate_examples_v6.py` - Generate Popper examples with complement predicates
- `bk.pl` - Background knowledge (predicate definitions)
- `bias.pl` - Search space bias (predicate declarations)
- `exs.pl` - Training examples

## Requirements

- Python 3.8+
- popper-ilp (`pip install popper-ilp`)
