# Plan: Spec-Only Trace Finding via Automaton (B) and SMT (C)

## Problem Statement

We want to replace Issy synthesis (~15 min per spec) with a faster method to find
**one satisfying trace** for a given TSL spec. The key requirement: the trace finder
must be **purely spec-driven** — no explicit game-rule encoding. It should derive
its transition system from the same TSLMT spec that Issy receives, and produce
traces that can then be evaluated against the actual game (C harness) to determine
win/loss. If the mined spec doesn't say "avoid holes," the trace should be free to
walk through holes — just like a synthesized controller would.

**What Issy receives** (from `spec_generator.py`):
```
var Int x
var Int y

goalx = i3();  goaly = i3();  hole0x = i1();  hole0y = i1();
BOUND_MIN = i0();  BOUND_MAX = i3();

inBounds = (gte x BOUND_MIN) && (lte x BOUND_MAX) && ...;
xMoves = [x <- x] || [x <- add x i1()] || [x <- sub x i1()];
yMoves = [y <- y] || [y <- add y i1()] || [y <- sub y i1()];

assume { eq x startx; eq y starty; }
guarantee {
    G inBounds;
    G ((xMoves && [y <- y]) || ([x <- x] && yMoves));
    <mined objective>;
}
```

Issy finds a **universal strategy** (winning against all plays) — 2EXPTIME.
We only need **one satisfying trace** — PSPACE (LTL satisfiability).

**What the trace finder does NOT know**: holes are fatal, goal is terminal, etc.
That's the C harness's job (game evaluation), which runs separately afterward.


## Architecture Overview

```
                     ┌─────────────────────────────┐
                     │    spec_generator.py         │
                     │  (already exists, unchanged) │
                     └─────────────┬───────────────┘
                                   │ TSLMT spec string
                                   ▼
                     ┌─────────────────────────────┐
                     │   TSLMT Parser (NEW)         │
                     │  extract variables, consts,  │
                     │  initial state, movement     │
                     │  rules, temporal objective   │
                     └─────────────┬───────────────┘
                                   │ structured spec data
                          ┌────────┴────────┐
                          ▼                 ▼
                ┌──────────────┐   ┌──────────────┐
                │  Option B    │   │  Option C    │
                │  spot/auto-  │   │  Z3/SMT      │
                │  maton BFS   │   │  bounded MC  │
                └──────┬───────┘   └──────┬───────┘
                       │                  │
                       └────────┬─────────┘
                                │ candidate trace
                                ▼
                     ┌─────────────────────────────┐
                     │   Game Evaluator             │
                     │  (reuse C harness or Python  │
                     │   equivalent — existing      │
                     │   infrastructure)             │
                     └─────────────┬───────────────┘
                                   │ SynthResult (success, trace)
                                   ▼
                              loop.py
```

Three layers:
1. **TSLMT Parser** — shared, extracts structured data from the generated spec
2. **Trace Finder** — Option B or C, finds one trace satisfying the spec
3. **Game Evaluator** — classifies trace as positive/negative (existing infra)


## Shared Component: TSLMT Parser

### Purpose

Parse the TSLMT spec string (output of `spec_generator.py`) into structured data
that both Option B and C can consume. This is the ONLY source of "game knowledge" —
everything comes from the spec, nothing hard-coded.

### What to Extract

```python
@dataclass
class TSLMTSpec:
    variables: list[tuple[str, str]]          # [("x", "Int"), ("y", "Int")]
    constants: dict[str, int]                  # {"goalx": 3, "hole0x": 1, ...}
    initial_state: dict[str, int|bool]         # {"x": 0, "y": 0}
    bounds: dict[str, tuple[int, int]]         # {"x": (0, 3), "y": (0, 3)}
    update_groups: list[dict[str, list[str]]]  # movement rule options (see below)
    objective: str                             # the mined temporal formula
    extra_guarantees: list[str]                # other guarantee clauses (e.g. taxi passenger logic)
```

### Movement Rule Extraction

The guarantee `G ((xMoves && [y <- y]) || ([x <- x] && yMoves))` encodes a
**disjunction of update combinations**. Each disjunct is a conjunction of
updates, one per variable.

Parsed into `update_groups`:
```python
[
    # Option 1: x moves, y stays
    {"x": ["x", "add x i1()", "sub x i1()"], "y": ["y"]},
    # Option 2: x stays, y moves
    {"x": ["x"], "y": ["y", "add y i1()", "sub y i1()"]},
]
```

Each group is a valid "move shape." At each timestep, the system picks one group,
then picks one update per variable from that group.

For Taxi, there's an additional variable `passengerInTaxi` with its own update
rule in a separate guarantee clause. The parser extracts this into `extra_guarantees`
or merges it into `update_groups`.

### Update Function Evaluation

We need to evaluate update functions on concrete values:
- `add x i1()` with x=2 → 3
- `sub x i1()` with x=2 → 1
- `x` (identity) with x=2 → 2

This requires a small evaluator for the function terms: `add`, `sub`, `id`,
`i<N>()` (integer literal). These are the only functions used in our specs.

```python
def eval_update(update_expr: str, state: dict[str, int]) -> int:
    """Evaluate an update expression like 'add x i1()' given current state."""
```

### Concrete Transition System

From `update_groups` + `bounds` + `initial_state`, enumerate all reachable states
and their transitions:

```python
def build_transitions(spec: TSLMTSpec) -> dict[State, list[tuple[Updates, State]]]:
    """
    Build explicit transition relation from spec.

    For each state, try each update_group, then each combination of updates
    within that group. Apply the updates to get the next state. Filter by bounds.

    Returns: mapping from state → list of (chosen_updates, next_state)
    """
```

For a 4x4 grid: 16 states, each with ~8 transitions (4 directions + identity combos).
For Taxi 5x5 with boolean: 50 states. All tiny — BFS is instant.

### File: `src/rl/tslmt_parser.py` (NEW)

```python
class TSLMTParser:
    """Parse a TSLMT spec string into structured data."""

    def parse(self, spec_text: str) -> TSLMTSpec: ...

    def _parse_variables(self, text: str) -> list[tuple[str, str]]: ...
    def _parse_constants(self, text: str) -> dict[str, int]: ...
    def _parse_assume(self, text: str) -> dict[str, int|bool]: ...
    def _parse_guarantees(self, text: str) -> tuple[list[dict], str, list[str]]: ...
    #                                               updates   obj  extras

def eval_update(expr: str, state: dict[str, int|bool]) -> int|bool: ...

def build_transition_system(spec: TSLMTSpec) -> TransitionSystem: ...

@dataclass
class TransitionSystem:
    initial: dict[str, int|bool]
    states: set[tuple]                              # all reachable states as tuples
    transitions: dict[tuple, list[tuple]]           # state → list of next states
    constants: dict[str, int]                       # for predicate evaluation
    variable_names: list[str]                       # ordered variable names
```


---


## Option B: Automaton-Based Trace Finding (spot)

### Core Idea

1. Convert the temporal objective to a **finite-trace automaton** using `spot`
2. Do **product BFS** over `(automaton_state × concrete_state)`
3. When the product reaches an accepting automaton state, reconstruct the trace

### Why Finite Traces?

Our traces are finite (games terminate). LTL over finite traces (LTLf) has
different semantics than standard LTL:
- `G(p)` = p holds at every step of the finite trace
- `F(p)` = p holds at some step of the finite trace

`spot` supports finite-trace translation via the `--finite` flag or the
`spot.translate(formula, 'finite')` Python API.

### Algorithm

```
Input: TSLMTSpec (from parser), temporal objective formula
Output: list[dict] (trace) or None

1. Parse the objective formula string
2. Build concrete transition system from TSLMTSpec
3. Translate temporal formula to finite-trace DFA using spot
   - Abstract propositions: each unique predicate (e.g., "eq_x_goalx") is an AP
   - spot.translate("F(eq_x_goalx & eq_y_goaly) & G(!eq_x_hole0x | !eq_y_hole0y)")
4. Product BFS:
   a. Start at (automaton_initial, concrete_initial)
   b. For each product state (q_aut, s_concrete):
      - Evaluate all APs on s_concrete → get a boolean assignment
      - Determine automaton transitions enabled by this assignment
      - For each concrete successor s' of s_concrete:
        - Evaluate APs on s' (for the NEXT state, needed for X handling)
        - For each enabled automaton transition → new product state (q_aut', s')
   c. If q_aut' is accepting, reconstruct trace from BFS parent map
   d. If BFS exhausted, return None (spec is unsatisfiable on this board)
```

### Predicate-to-AP Mapping

The temporal formula uses predicates like `eq x goalx`. For spot, we need
atomic proposition names. Strategy:

```python
def formula_to_spot(op: Op, constants: dict) -> tuple[str, dict[str, Callable]]:
    """
    Convert Op tree to spot-compatible LTL string + AP evaluator map.

    Returns:
        spot_formula: "F(p0 & p1) & G(!p2 | !p3)"
        ap_evaluators: {"p0": lambda s: s["x"] == 3, "p1": lambda s: s["y"] == 3, ...}
    """
```

Each unique state predicate becomes a named proposition (p0, p1, ...).
The evaluator map lets us check which propositions hold at each concrete state.

### Handling X (Next) in the Formula

`spot` natively handles X. In the product BFS, when the automaton takes a
transition that requires evaluating a proposition "at the next step," we
evaluate it on the concrete successor state. This is natural in the product
construction — the automaton transition reads the current labeling, and X
shifts the evaluation to the next step.

### Handling Updates in the Formula

Some mined specs contain Update terms: `G([x <- add x i1()] || [x <- x])`.
These can't be directly encoded as spot APs because they constrain which
transition was taken, not the state.

Strategy: convert Updates to predicates on (current, next) state:
- `[x <- add x i1()]` becomes `next_x == current_x + 1`
- `[x <- x]` becomes `next_x == current_x`

In the product BFS, when expanding a concrete transition (s → s'), we know
both current and next state, so we can evaluate these.

If the spec contains Updates that interact with the transition system's own
update rules (beyond what movement rules already define), this gets complex.
For the MVP, specs with Updates that conflict with or duplicate movement rules
could fall back to synthesis.

### File: `src/rl/trace_finder_automaton.py` (NEW)

```python
import spot

class AutomatonTraceFinder:
    """Find satisfying traces via spot automaton + product BFS."""

    def __init__(self, transition_system: TransitionSystem):
        self.ts = transition_system

    def find_trace(self, objective: str, max_steps: int = 200) -> list[dict] | None:
        """
        Find one trace satisfying the objective.

        Args:
            objective: temporal formula string (the mined objective)
            max_steps: maximum trace length

        Returns:
            List of state dicts (the trace) or None if unsatisfiable
        """

    def _formula_to_spot(self, op: Op) -> tuple[spot.formula, dict[str, Callable]]:
        """Convert Op tree to spot formula + AP evaluators."""

    def _product_bfs(self, aut, ap_evals, max_steps) -> list[tuple] | None:
        """BFS over (automaton_state, concrete_state) product."""

    def _reconstruct(self, goal, parents) -> list[dict]:
        """Reconstruct trace from BFS parent map."""
```

### Dependencies

- `spot` Python package: `pip install spot` (or build from source)
- spot is well-maintained (LRDE/EPITA), has Python wheels for Linux/Mac
- If spot is unavailable, fall back to Option C or synthesis


---


## Option C: SMT-Based Bounded Model Checking (Z3)

### Core Idea

"Unroll" the spec for k timesteps: create a copy of every variable at each
step, replace temporal operators with quantifiers over step indices, encode
movement constraints as per-step relations, and let Z3 find a satisfying
assignment.

### What "Unrolling" Means — Concrete Example

**Spec**: `F(eq x goalx && eq y goaly) && G(!(eq x hole0x && eq y hole0y))`
**Params**: grid=4x4, goal=(3,3), hole0=(1,1), start=(0,0)
**Bound**: k=10

**Z3 variables**: `x_0, y_0, x_1, y_1, ..., x_10, y_10`

```python
from z3 import *

k = 10
x = [Int(f'x_{i}') for i in range(k + 1)]
y = [Int(f'y_{i}') for i in range(k + 1)]

s = Solver()

# ─── Initial state ───
s.add(x[0] == 0, y[0] == 0)

# ─── Bounds: G(inBounds) ───
for i in range(k + 1):
    s.add(x[i] >= 0, x[i] <= 3)
    s.add(y[i] >= 0, y[i] <= 3)

# ─── Movement: G((xMoves && y stays) || (x stays && yMoves)) ───
for i in range(k):
    x_moves = And(Or(x[i+1] == x[i] - 1, x[i+1] == x[i], x[i+1] == x[i] + 1),
                  y[i+1] == y[i])
    y_moves = And(x[i+1] == x[i],
                  Or(y[i+1] == y[i] - 1, y[i+1] == y[i], y[i+1] == y[i] + 1))
    s.add(Or(x_moves, y_moves))

# ─── G(!(eq x 1 && eq y 1)): at every step, not at hole ───
for i in range(k + 1):
    s.add(Not(And(x[i] == 1, y[i] == 1)))

# ─── F(eq x 3 && eq y 3): at some step, at goal ───
s.add(Or(*[And(x[i] == 3, y[i] == 3) for i in range(k + 1)]))

# ─── Solve ───
if s.check() == sat:
    m = s.model()
    trace = [{"x": m[x[i]].as_long(), "y": m[y[i]].as_long()} for i in range(k + 1)]
```

### Temporal Operator Unrolling Rules

Given bound k, variables at each step i, here's how each temporal operator
translates:

```
┌──────────────────────────┬────────────────────────────────────────────────────┐
│ Temporal Formula         │ Unrolled to Z3 (for steps 0..k)                   │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ G(φ)                     │ ∧_{i=0}^{k} φ(i)                                 │
│                          │ (φ holds at every step)                            │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ F(φ)                     │ ∨_{i=0}^{k} φ(i)                                 │
│                          │ (φ holds at some step)                             │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ X(φ)  at step i          │ φ(i+1)                                            │
│                          │ (φ holds at the next step)                         │
│                          │ At step k: False (no next step in finite trace)    │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ φ U ψ                    │ ∨_{j=0}^{k} (ψ(j) ∧ ∧_{i=0}^{j-1} φ(i))        │
│                          │ (ψ holds at some step j, φ holds at all prior)    │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ F(φ ∧ F(ψ))             │ ∨_{j=0}^{k} (φ(j) ∧ ∨_{m=j+1}^{k} ψ(m))        │
│ (multi-stage)            │ (φ at step j, then ψ at some later step m)        │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ G(φ → X ψ)              │ ∧_{i=0}^{k-1} (φ(i) → ψ(i+1))                   │
│ (edge constraint)        │ (if φ now, then ψ next — for all steps)           │
└──────────────────────────┴────────────────────────────────────────────────────┘
```

### Recursive Unroller

```python
def unroll(op: Op, step: int, k: int, vars: StepVars) -> z3.BoolRef:
    """
    Recursively unroll a temporal formula at a given step.

    Args:
        op: the Op node
        step: current timestep index
        k: maximum bound
        vars: mapping from (var_name, step_index) → z3 variable

    Returns:
        Z3 boolean expression
    """
    match op:
        case Predicate(pred, inputs):
            # Evaluate predicate on step-indexed variables
            args = [resolve_var(v, step, vars) for v in inputs]
            return pred_to_z3(pred.name, args)

        case BooleanAP(name):
            return vars.get_bool(name, step)

        case Not(inner):
            return z3.Not(unroll(inner, step, k, vars))

        case And(left, right):
            return z3.And(unroll(left, step, k, vars), unroll(right, step, k, vars))

        case Or(left, right):
            return z3.Or(unroll(left, step, k, vars), unroll(right, step, k, vars))

        case Always(inner):
            return z3.And(*[unroll(inner, i, k, vars) for i in range(step, k + 1)])

        case Eventually(inner):
            return z3.Or(*[unroll(inner, i, k, vars) for i in range(step, k + 1)])

        case Next(inner):
            if step >= k:
                return z3.BoolVal(False)  # no next step (finite trace)
            return unroll(inner, step + 1, k, vars)

        case Until(left, right):
            # exists j >= step: right(j) and forall i in [step, j): left(i)
            clauses = []
            for j in range(step, k + 1):
                prefix = [unroll(left, i, k, vars) for i in range(step, j)]
                clauses.append(z3.And(*prefix, unroll(right, j, k, vars)) if prefix
                               else unroll(right, j, k, vars))
            return z3.Or(*clauses)

        case Update(var, func, inputs):
            # [x <- add x i1()] at step i means: x_{i+1} == add(x_i, 1)
            # This constrains the transition, not just the state
            if step >= k:
                return z3.BoolVal(True)  # vacuously true at last step
            current_args = [resolve_var(v, step, vars) for v in inputs]
            result = func_to_z3(func.name, current_args)
            next_var = vars.get(var.name, step + 1)
            return next_var == result
```

### Handling the Bound k

The bound k is the maximum trace length. If Z3 returns UNSAT, the spec might
still be satisfiable with a longer trace. Strategy:

```python
def find_trace_smt(spec: TSLMTSpec, objective: Op, max_k: int = 50) -> list[dict] | None:
    for k in [10, 20, 30, max_k]:  # iterative deepening
        result = try_bound(spec, objective, k)
        if result is not None:
            return result
    return None  # unsatisfiable within max_k steps
```

Alternatively, start with a small k and double until SAT or max reached.

For our games, traces are short (grid paths are at most ~25 steps for a 5x5
grid), so k=50 is more than enough.

### Handling Updates in the Objective

If the mined objective contains Updates like `G([x <- add x i1()] || [x <- x])`,
these constrain which transitions are taken. The unroller converts them to
constraints on consecutive step variables:

- `[x <- add x i1()]` at step i → `x_{i+1} == x_i + 1`
- `[x <- x]` at step i → `x_{i+1} == x_i`

These are natural Z3 integer constraints — no special handling needed.

### File: `src/rl/trace_finder_smt.py` (NEW)

```python
from z3 import *

class SMTTraceFinder:
    """Find satisfying traces via Z3 bounded model checking."""

    def __init__(self, transition_system: TransitionSystem):
        self.ts = transition_system

    def find_trace(self, objective: Op, max_steps: int = 50) -> list[dict] | None:
        """
        Find one trace satisfying the objective via bounded model checking.

        Tries increasing bounds until SAT or max_steps reached.
        """

    def _try_bound(self, objective: Op, k: int) -> list[dict] | None:
        """Try to find a trace of length exactly k."""

    def _create_step_vars(self, k: int) -> StepVars:
        """Create Z3 variables for each (variable, timestep) pair."""

    def _add_initial_state(self, solver: Solver, vars: StepVars): ...
    def _add_movement_constraints(self, solver: Solver, vars: StepVars, k: int): ...
    def _add_bounds_constraints(self, solver: Solver, vars: StepVars, k: int): ...

    def _unroll(self, op: Op, step: int, k: int, vars: StepVars) -> BoolRef:
        """Recursively unroll temporal formula (see algorithm above)."""

    def _extract_trace(self, model: ModelRef, vars: StepVars, k: int) -> list[dict]:
        """Extract concrete trace from Z3 model."""
```

### Dependencies

- `z3-solver`: `pip install z3-solver`
- Widely available, well-maintained, pip-installable


---


## Game Evaluation: Classifying Traces as Positive/Negative

### The Problem

The trace finder produces a trace satisfying the SPEC. But the spec doesn't
encode all game rules (holes are fatal, cliffs kill you, etc.). We need to
evaluate the trace against the actual game to determine if it's a winning or
losing trace — exactly like the C harness does for synthesized controllers.

### Example

Spec: `F(eq x goalx && eq y goaly)` (no hole avoidance)
Trace found: `(0,0) → (0,1) → (1,1) → (2,1) → ... → (3,3)`

This trace satisfies the spec (reaches the goal). But at step 2, it passes
through (1,1) which is a hole. In the actual game, the agent would die at
step 2. So this is a **negative trace** (truncated to 3 steps: `(0,0) → (0,1) → (1,1)`).

This is EXACTLY what happens with synthesis: if the synthesized controller walks
into a hole, the C harness kills it → negative trace.

### Approach: Reuse Existing C Harness

The cleanest approach: build a "replay" controller that follows the found trace,
embed it in the game harness, and run it. The harness checks all game rules.

```python
def evaluate_trace_via_harness(
    trace: list[dict],
    game: str,
    params: dict,
    api: SynthesisAPI,
) -> SynthResult:
    """
    Evaluate a candidate trace against the actual game.

    Generates a C controller that replays the trace step-by-step,
    embeds it in the game harness, and runs it. The harness detects
    if/when the trace violates game rules (hole, cliff, etc.).

    Returns SynthResult with:
    - success=True if trace completed without game violation
    - success=False + truncated trace if game terminated early
    """
    # Generate replay controller:
    # int step = 0;
    # int trace_x[] = {0, 0, 1, 2, ...};
    # int trace_y[] = {0, 1, 1, 1, ...};
    # int main() {
    #     x = trace_x[0]; y = trace_y[0];
    #     while(1) {
    #         read_inputs();  // ← harness checks holes/cliffs/goal
    #         step++;
    #         if (step >= TRACE_LEN) exit(3);  // timeout
    #         x = trace_x[step]; y = trace_y[step];
    #     }
    # }
    replay_code = generate_replay_controller(trace, game)
    run_result, actual_trace = api.run_controller(params, replay_code)
    return SynthResult(
        success=run_result.success,
        trace=actual_trace,
        steps=run_result.steps,
        error_message=run_result.error_message if not run_result.success else None,
    )
```

**Advantages**:
- Reuses existing game harness infrastructure
- Zero game-specific Python code
- Guaranteed to match synthesis behavior exactly
- If game rules change, only the C harness needs updating

**Disadvantages**:
- Requires compiling and running C code for each trace (~50ms overhead)
- For the RL loop this is fine (trace finding itself is <1ms, so 50ms total is still
  orders of magnitude faster than 15-minute synthesis)

### Alternative: Python Game Evaluator

If C compilation overhead is unacceptable (unlikely), a Python evaluator
could check game-specific terminal conditions. But this duplicates game
rules and risks diverging from the C harness.

We could also skip game evaluation entirely for some use cases — if the
mined spec is strong enough to encode all relevant safety, the trace will
be valid. But in general we need it, especially for weak/partial specs.


---


## Integration with loop.py

### API Changes to `synt.py`

Add a `find_trace()` method to `SynthesisAPI`:

```python
def find_trace(
    self,
    objective: str,
    params: dict[str, Any],
    max_steps: int = 200,
    method: str = "smt",  # "smt" or "automaton"
) -> SynthResult:
    """
    Find a single satisfying trace, then evaluate it against the game.

    1. Generate TSLMT spec (same as for synthesis)
    2. Parse spec → structured data
    3. Build transition system
    4. Find trace via automaton or SMT
    5. Evaluate trace against game harness
    6. Return SynthResult (same format as synthesize_and_run)
    """
    # Step 1-2: Generate and parse spec
    tslmt_text = self.generate_tslmt_spec(params, objective)
    spec_data = TSLMTParser().parse(tslmt_text)

    # Step 3: Build transition system
    ts = build_transition_system(spec_data)

    # Step 4: Find trace
    objective_op = parse_tsl(objective)
    if method == "automaton":
        finder = AutomatonTraceFinder(ts)
    else:
        finder = SMTTraceFinder(ts)

    candidate_trace = finder.find_trace(objective_op, max_steps=max_steps)

    if candidate_trace is None:
        return SynthResult(success=False, error_message="No satisfying trace found")

    # Step 5: Evaluate against game
    return evaluate_trace_via_harness(candidate_trace, self.game, params, self)
```

### Changes to `loop.py`

Minimal — same as previous plan:

```python
# In rollout():
if self.use_trace_finder:
    result = self.api.find_trace(
        objective=str(objective), params=self.params,
        max_steps=200, method=self.trace_finder_method
    )
else:
    result = self.api.synthesize_and_run(
        objective=str(objective), params=self.params, timeout_steps=20
    )
```

CLI flags:
```
--rollout-mode {trace-finder, synthesis}   (default: trace-finder)
--trace-method {smt, automaton}            (default: smt)
```


---


## Comparison: Option B vs Option C

```
┌────────────────────────┬─────────────────────────┬──────────────────────────┐
│                        │ Option B (spot/automaton)│ Option C (Z3/SMT)        │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Theory                 │ LTL → automaton →       │ Bounded model checking   │
│                        │ product BFS             │ (temporal → constraints) │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Completeness           │ Complete for any trace  │ Complete up to bound k   │
│                        │ length (automaton is    │ (may miss longer traces) │
│                        │ finite, BFS explores    │                          │
│                        │ all reachable states)   │                          │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Speed                  │ ~1-10ms (small state    │ ~1-100ms (depends on k   │
│                        │ space, small formulas)  │ and formula complexity)   │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Handles Updates        │ Needs special handling  │ Natural (Updates become  │
│ in objective           │ (encode as transition   │ constraints on x_i vs    │
│                        │ predicates)             │ x_{i+1})                 │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Handles Until          │ Native (automaton       │ Unrolling is correct but │
│                        │ construction handles U) │ produces large formulas  │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Dependencies           │ spot (may need build    │ z3-solver (pip install)  │
│                        │ from source on some OS) │                          │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Implementation         │ Medium (automaton API,  │ Lower (Z3 API is         │
│ complexity             │ product construction,   │ straightforward, unroll  │
│                        │ AP mapping)             │ is mechanical)           │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Debugging              │ Can visualize automaton │ Can inspect Z3 model     │
│                        │ (spot has dot export)   │ directly                 │
├────────────────────────┼─────────────────────────┼──────────────────────────┤
│ Extensibility          │ Full LTL, could extend  │ Easy to add new          │
│                        │ to LTL+past             │ constraint types         │
└────────────────────────┴─────────────────────────┴──────────────────────────┘
```

### Recommendation

**Start with Option C (Z3)**:
- Easier to implement and debug
- pip-installable dependency
- Handles Updates naturally (our mined specs often contain them)
- For our small state spaces and short traces, bound limitations don't matter

**Add Option B (spot) afterward**:
- Provides completeness guarantee (no bound guessing)
- Better for larger state spaces where explicit BFS is still fast but Z3 struggles
- More principled from a formal methods perspective

Both share the same `TSLMTParser` and `TransitionSystem` infrastructure +
the same game evaluation step.


---


## Implementation Order

1. `src/rl/tslmt_parser.py` — TSLMT spec parser + transition system builder
2. `src/rl/trace_finder_smt.py` — Z3-based trace finder (Option C)
3. Game evaluation via C harness replay (in `synt.py`)
4. Integration into `synt.py` and `loop.py`
5. `src/rl/trace_finder_automaton.py` — spot-based trace finder (Option B)
6. Tests and validation against synthesis results


---


## Open Questions

1. **Taxi passenger logic**: The guarantee clause for `passengerInTaxi` updates
   is complex (`G(([passengerInTaxi <- true] && ...) || ([passengerInTaxi <- false] && ...))`).
   Should the TSLMT parser handle this generically, or special-case it?
   → Prefer generic: any G-wrapped disjunction of update conjunctions gets
   merged into `update_groups`.

2. **Blackjack**: Has `inp` (environment input) variables (`dealer`, `card`).
   These are non-deterministic — the trace finder would need to choose values
   for them. Both B and C can handle this (existential quantification over
   inputs), but it adds complexity. Defer for now.

3. **Iterative deepening strategy for Option C**: Start at k=10 and double?
   Or use binary search? For our games, k=grid_size*grid_size is a safe upper
   bound (longest non-repeating path).

4. **Trace diversity**: Even with the correct architecture, the trace finder
   is deterministic (Z3/BFS always returns the same trace for the same spec).
   For the RL loop to learn, we may need diversity. Options:
   - Add random soft constraints to Z3 (`s.add_soft(x[3] == random_val)`)
   - Shuffle BFS successor order
   - Exclude previously-found traces as blocking clauses
   This is a separate concern from correctness — address after the basic
   implementation works.
