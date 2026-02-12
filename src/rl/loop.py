import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from src.bolt.log2tslf import Update, Predicate, BooleanAP
from synt import SynthesisAPI
from toytext import config_to_params, generate_config, get_goal_spec
from parser import Op, And, Or, Not, Next, Until, Eventually, Always, neg, parse_tsl
from src.bolt.log2tslf import Update, Predicate, BooleanAP
from games.synt.spec_transformer import transform_spec



class SpecPool:
    def __init__(self):
        # self.spec: Optional[Op] = None
        self.stack: list[Op] = []
        self.visited: set[Op] = set()

    def _collect_pred_update(self, op: Op) -> tuple[set, set]:
        """
        Recursively collect unique predicates and updates in an Op tree.
        Returns (predicate_set, update_set).
        """
        match op:
            case Predicate():
                return ({op}, set())
            case Update():
                return (set(), {op})
            case BooleanAP():
                return (set(), set())
            case Not(inner):
                return self._collect_pred_update(inner)
            case And(left, right) | Or(left, right) | Until(left, right):
                lp, lu = self._collect_pred_update(left)
                rp, ru = self._collect_pred_update(right)
                return (lp | rp, lu | ru)
            case Next(inner) | Eventually(inner) | Always(inner):
                return self._collect_pred_update(inner)
            case _:
                return (set(), set())

    def pop_by_pred(self) -> Optional[Op]:
        """
        Pop the spec with most predicates, then fewest updates.
        This prioritizes semantically richer specs over trivial update-only specs.
        """
        if not self.stack:
            print(f"  [SpecPool.pop_by_pred] STACK EMPTY (visited size: {len(self.visited)})")
            return None

        # Rank by (unique predicates DESC, unique updates ASC)
        def rank_key(spec: Op) -> tuple[int, int]:
            preds, updates = self._collect_pred_update(spec)
            return (-len(preds), len(updates))

        # Find best spec
        best_spec = min(self.stack, key=rank_key)
        best_preds, best_updates = self._collect_pred_update(best_spec)

        # Remove from stack and add to visited
        self.stack.remove(best_spec)
        self.visited.add(best_spec)

        print(f"  [SpecPool.pop_by_pred] POPPED: {best_spec} (unique_preds={len(best_preds)}, unique_updates={len(best_updates)}, visited size now: {len(self.visited)})")
        return best_spec

    def add(self, spec: Op | list[Op]) -> bool:
        added = False
        if isinstance(spec, list):
            # TODO: maybe change this to simply adopt the new set of specs instead of appending
            for s in spec:
                if s in self.visited:
                    print(f"  [SpecPool.add] REJECTED (visited): {s}")
                elif s in self.stack:
                    print(f"  [SpecPool.add] REJECTED (in stack): {s}")
                else:
                    print(f"  [SpecPool.add] ADDED: {s}")
                    self.stack.append(s)
                    added = True
        else:
            if spec in self.visited:
                print(f"  [SpecPool.add] REJECTED (visited): {spec}")
            elif spec in self.stack:
                print(f"  [SpecPool.add] REJECTED (in stack): {spec}")
            else:
                print(f"  [SpecPool.add] ADDED: {spec}")
                self.stack.append(spec)
                added = True
        return added
    
    # def visit(self, spec: Op) -> bool:
    #     if spec in self.visited:
    #         return False
    #     self.visited.add(spec)
    #     return True

    def pop(self, rank: str | None = None) -> Optional[Op]:
        """
        Pop a spec from the pool.

        Args:
            rank: Ranking strategy. If "pred", uses pop_by_pred (most predicates, fewest updates).
                  If None, uses LIFO (last in, first out).
        """
        if rank == "pred":
            return self.pop_by_pred()

        # Default: LIFO behavior
        spec = None
        if self.stack:
            spec = self.stack.pop(-1)
            self.visited.add(spec)
            print(f"  [SpecPool.pop] POPPED (LIFO): {spec} (visited size now: {len(self.visited)})")
        else:
            print(f"  [SpecPool.pop] STACK EMPTY (visited size: {len(self.visited)})")
        return spec
    

class RLLoop:
    def __init__(self, 
                 game: str, 
                 varied: bool = False, 
                 work_dir: Optional[Path] = None,
                 goal: None | Optional[Op] = None):
        self.game = game
        self.varied = varied
        self.goal  = goal # goal spec, optionally provided
        self.api = SynthesisAPI(game=game, synthesis_timeout_minutes=20)

        # Setup working directory (always absolute to avoid cwd issues)
        self.work_dir = Path(work_dir).resolve() if work_dir else Path(tempfile.mkdtemp(prefix="rl_loop_"))
        self.pos_dir = self.work_dir / "pos"
        self.neg_dir = self.work_dir / "neg"
        self.out_dir = self.work_dir / "out"

        # Clean old traces before starting
        for d in [self.pos_dir, self.neg_dir, self.out_dir]:
            if d.exists():
                shutil.rmtree(d)

        self.pos_dir.mkdir(parents=True, exist_ok=True)
        self.neg_dir.mkdir(parents=True, exist_ok=True)

        # Generate ONE fixed board config for the entire loop
        self.config = generate_config(self.game, varied=self.varied)
        self.params = config_to_params(self.config, self.game)

        # DEBUG: show the fixed board config
        # goal = self.params.get("goal", {})
        # holes = self.params.get("holes", [])
        # print(f"Fixed board config: goal=({goal.get('x')},{goal.get('y')}), holes={[(h.get('x'),h.get('y')) for h in holes]}")

        # Trace storage
        self.pos_traces: list[list[dict]] = []
        self.neg_traces: list[list[dict]] = []

        # Spec
        self.spec: Optional[Op] = None
        self.specs: SpecPool = SpecPool()
        self._traces_at_last_collect = 0
        # self.spec_queue : set[Op] = set()
        # self.spec_visited: set[Op] = set()
        # self.spec_queue: SpecQueue = SpecQueue()

        # print("Performing initial random rollouts.")
        # Do multiple random rollouts to increase chance of positive trace
        # num_initial_rollouts = 5
        # for i in range(num_initial_rollouts):
        #     print(f"Initial rollout {i+1}/{num_initial_rollouts}")
        #     self.rollout()  # will do random rollout as no spec yet
        #     if self.pos_traces:
        #         print(f"Got positive trace on rollout {i+1}, stopping initial rollouts.")
        #         break

        # self._random_rollout() # NOTE: probably can change to just rollout

        # Refine policy from collected traces
        # self.refine_policy()

        # exit(0)

        

    def _get_default_liveness(self) -> Optional[Op]:
        """Get default liveness goal for the game."""
        if self.game in ("frozen_lake", "ice_lake"):
            # F (eq x goalx && eq y goaly) - use constant names, not literal values
            from parser import Eventually, And
            from src.bolt.log2tslf import Predicate
            # Create predicates using constant names (goalx, goaly defined in TSLMT spec)
            eq_x = Predicate.from_string("eq x goalx")
            eq_y = Predicate.from_string("eq y goaly")
            return Eventually(And(eq_x, eq_y))
        return None

    def _random_rollout(self, continue_on_timeout: bool = False) -> bool:
        """Execute one random rollout on the fixed board config."""
        result = self.api.initial_random_run(params=self.params, timeout_steps=20)
        print(f"Random rollout: {'Positive' if result.is_positive else 'Negative'}, {len(result.trace)} steps")
        if result.failure_reason:
            print(f"  Failure reason: {result.failure_reason}")

        # Continue with random walk if timeout and flag is set
        if continue_on_timeout and self._is_timeout_failure(result.failure_reason) and result.trace:
            print("Timeout detected, continuing with random rollout...")
            return self._continue_with_random(result.trace)

        return self._store_trace(result.trace, result.is_positive, self.params,
                                 failure_reason=result.failure_reason)

    def _is_timeout_failure(self, failure_reason: Optional[str]) -> bool:
        """Check if the failure reason indicates a timeout (too many steps)."""
        if not failure_reason:
            return False
        reason_lower = failure_reason.lower()
        return "timeout" in reason_lower or "step limit" in reason_lower

    def _store_trace(self, trace: list[dict], is_positive: bool, params: dict,
                     failure_reason: Optional[str] = None) -> bool:
        """Store trace in appropriate directory.

        Args:
            trace: The trace to store
            is_positive: Whether this is a positive trace
            params: Game parameters for constant extraction
            failure_reason: Why the trace failed (if negative)

        Returns:
            True if positive, False if negative (or skipped)
        """
        if not trace:
            print(f"Warning: Empty trace, skipping storage")
            return is_positive

        # Skip timeout traces for negative examples - they're not useful for learning
        if not is_positive and self._is_timeout_failure(failure_reason):
            print(f"Skipping timeout trace (not adding to negatives): {failure_reason}")
            return False

        constants = self._get_constants(params)
        # Add constants to each entry for in-memory storage too
        augmented_trace = [{**entry, **constants} for entry in trace]

        if is_positive:
            self.pos_traces.append(augmented_trace)
            path = self.pos_dir / f"pos_{len(self.pos_traces)}.jsonl"
        else:
            self.neg_traces.append(augmented_trace)
            path = self.neg_dir / f"neg_{len(self.neg_traces)}.jsonl"

        with open(path, "w") as f:
            for entry in augmented_trace:
                f.write(json.dumps(entry) + "\n")
        return is_positive

    def _continue_with_random(self, partial_trace: list[dict], timeout_steps: int = 50) -> bool:
        """
        Continue a timed-out trace with random rollout until win or lose.

        Args:
            partial_trace: The trace from the timed-out spec rollout
            timeout_steps: Max steps for the random continuation

        Returns:
            True if the combined trace is positive, False otherwise
        """
        if not partial_trace:
            print("Cannot continue empty trace")
            return False

        final_state = partial_trace[-1]
        print(f"Continuing from state: {final_state}")

        # Random walk from final state
        continuation = self.api.initial_random_run(
            params=self.params,
            timeout_steps=timeout_steps,
            start_state=final_state,
        )

        # Combine traces (skip first entry of continuation to avoid duplicate)
        combined_trace = partial_trace + continuation.trace[1:]

        print(f"Continuation: {'Positive' if continuation.is_positive else 'Negative'}, "
              f"{len(continuation.trace)} steps, combined total: {len(combined_trace)} steps")

        return self._store_trace(
            combined_trace,
            continuation.is_positive,
            self.params,
            failure_reason=continuation.failure_reason,
        )

    def _get_constants(self, params: dict) -> dict:
        """Extract constants from params for trace augmentation (as tuples)."""
        if self.game in ("ice_lake", "frozen_lake"):
            goal = params.get("goal", {})
            constants = {
                "goal": [goal.get("x", 3), goal.get("y", 3)],
            }
            # Add hole positions as tuples (hole0, hole1, hole2, ...)
            holes = params.get("holes", [])
            for i, hole in enumerate(holes):
                if isinstance(hole, dict):
                    constants[f"hole{i}"] = [hole.get("x", 0), hole.get("y", 0)]
                elif isinstance(hole, (list, tuple)):
                    constants[f"hole{i}"] = list(hole)
            return constants
        elif self.game == "cliff_walking":
            goal = params.get("goal_pos", {})
            return {"goal": [goal.get("x", 11), goal.get("y", 0)]}
        return {}

    def _run_mining(self,
                    mode: str = "safety-liveness",
                    max_size: int = 8,
                    opt: str = "--first-all",
                    pos_dir: Optional[Path] = None,
                    neg_dir: Optional[Path] = None) -> tuple[Optional[Op], list[Op]]:
        """
        Run mining subprocess with specified parameters.

        Args:
            mode: Mining mode (safety-liveness, liveness, safety)
            max_size: Maximum formula size
            opt: Search option (--first-all, --collect-all)
            pos_dir: Explicit path to positive traces directory
            neg_dir: Explicit path to negative traces directory

        Returns:
            Tuple of (primary_spec, all_specs) where all_specs includes alternatives
        """
        script = Path(__file__).parent.parent / "mine.sh"

        cmd = [
            "bash", str(script), str(self.work_dir), opt,
            "--mode", mode,
            "--game", self.game,
            "--max-size", str(max_size),
            "--prune", "--self-inputs-only"
        ]

        # Add explicit pos/neg directories if specified
        if pos_dir:
            cmd.extend(["--pos", str(pos_dir)])
        if neg_dir:
            cmd.extend(["--neg", str(neg_dir)])

        print(f"Mining with mode={mode}, max_size={max_size}...")
        print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                cwd=Path(__file__).parent.parent.parent,
                timeout=120  # 2 minute timeout
            )
            print(f"  Mining completed with return code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  Mining timed out after 120 seconds")
            return None, []

        if result.returncode != 0:
            print(f"Mining failed: {result.stderr}")
            return None, []

        # Helper to transform and parse a spec
        def transform_and_parse(text: str, is_safety: bool = False) -> Op | None:
            """Transform spec text to Issy format, then parse to Op."""
            transformed = transform_spec(text, is_safety=is_safety, game=self.game)
            return parse_tsl(transformed)

        # Parse specs - transform to Issy format BEFORE parsing
        # This way all stored specs are in transformed format (x, y instead of player[0], player[1])
        primary_spec = None
        liveness_file = self.out_dir / "liveness.tsl"
        safety_file = self.out_dir / "safety.tsl"

        liveness_spec = None
        safety_spec = None

        if liveness_file.exists():
            liveness_text = liveness_file.read_text().strip()
            if liveness_text:
                try:
                    liveness_spec = transform_and_parse(liveness_text, is_safety=False)
                except Exception as e:
                    print(f"Failed to parse liveness spec: {e}")

        if safety_file.exists():
            safety_text = safety_file.read_text().strip()
            if safety_text:
                try:
                    safety_spec = transform_and_parse(safety_text, is_safety=True)
                except Exception as e:
                    print(f"Failed to parse safety spec: {e}")

        # Combine liveness and safety if both exist
        # Track individual specs to exclude from queue
        primary_components = set()
        if liveness_spec and safety_spec:
            primary_spec = And(liveness_spec, safety_spec)
            primary_components.add(liveness_spec)
            primary_components.add(safety_spec)
        elif liveness_spec:
            primary_spec = liveness_spec
        elif safety_spec:
            primary_spec = safety_spec

        # Collect all alternative specs from all_liveness.tsl and all_safety.tsl
        # Exclude specs that are components of the primary combined spec
        all_specs: list[Op] = []
        for spec_list_file, is_safety in [(self.out_dir / "all_liveness.tsl", False),
                                           (self.out_dir / "all_safety.tsl", True)]:
            if spec_list_file.exists():
                for line in spec_list_file.read_text().strip().split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            spec = transform_and_parse(line, is_safety=is_safety)
                            # Skip if this spec is already part of the primary combined spec
                            if spec and spec not in all_specs and spec not in primary_components:
                                all_specs.append(spec)
                        except Exception as e:
                            print(f"Failed to parse spec '{line}': {e}")

        return primary_spec, all_specs

    def refine_policy(self,
                      mode: str = "safety-liveness",
                      max_size: int = 8,
                      opt: str = "--first-all") -> Optional[Op]:
        """
        Mine spec from current traces.

        If no positive traces exist, mines from negative traces as positive
        (liveness mode only), then negates the result.

        Populates spec_queue with all alternative specs for later inspection.
        """
        if not self.pos_traces and not self.neg_traces:
            print("No traces to mine from.")
            return None
        
        # if self.spec:
        #     self.spec_visited.add(self.spec)

        if not self.pos_traces:
            # No positive traces: flip pos/neg, mine liveness, negate result
            print("No positive traces. Mining liveness from negatives (flipped), then negating.")
            spec, all_specs = self._run_mining(
                mode="liveness",
                max_size=max_size,
                opt=opt,
                pos_dir=self.neg_dir,  # Flip: treat neg as pos
                neg_dir=self.pos_dir   # Flip: treat pos as neg (empty)
            )

            print(f"[refine_policy] Mining returned: primary={spec}, all_specs count={len(all_specs)}")
            if spec:
                # self.spec = neg(spec)
                # self.spec_queue.add(neg(spec))
                print(f"Negated primary spec: {neg(spec)}")
                print(f"[refine_policy] NOTE: Primary spec is NOT being added to stack!")
                # specss = reversed([neg(alt_spec) for alt_spec in all_specs])
                # Negate all alternative specs too
                negated_alts = list(reversed([neg(alt_spec) for alt_spec in all_specs]))
                print(f"[refine_policy] Adding {len(negated_alts)} negated alternatives:")

                added = self.specs.add(negated_alts)
                # if not added:
                #     print(f"[refine_policy] No new negated alternatives were added to the stack.")
                # for alt_spec in all_specs:
                #     negated = neg(alt_spec)
                    # if negated not in self.spec_visited:
                    #     self.spec_queue.add(negated)
            else:
                print(f"[refine_policy] No primary spec returned from mining")
        else:
            # Normal case: mine with actual pos/neg
            spec, all_specs = self._run_mining(
                mode="safety",
                max_size=max_size,
                opt=opt
            )
            print(f"[refine_policy] Mining returned: primary={spec}, all_specs count={len(all_specs)}")
            # Add alternatives to queue (excluding primary spec)
            # Filter: skip if visited or already in stack
            if spec:
                print(f"[refine_policy] NOTE: Primary spec is NOT being added to stack!")
                print(f"[refine_policy] Adding {len(all_specs)} alternatives:")
                self.specs.add(list(reversed(all_specs)))  # Add all specs to stack first
                # self.specs.add(spec)  # Add primary spec to stack last (so it's popped first)
                # self.spec_queue.add(spec)  # Add primary spec to queue first
                # # self.spec_queue.add(all_specs)  # Add all alternatives to queue
                # for alt_spec in all_specs:
                #     if alt_spec not in self.spec_visited:
                #         self.spec_queue.add(alt_spec)
            else:
                print(f"[refine_policy] No primary spec returned from mining")
        # Mark primary spec as visited
        print(f"Spec stack has {len(self.specs.stack)} alternative specs:")
        for alt_spec in self.specs.stack:
            print(f"  * {alt_spec}")

        return self.spec

    def rollout(self, spec, continue_on_timeout: bool = False) -> bool:
        """
        Synthesize agent from spec on the fixed board config.

        When self.goal is set, the spec from the stack is treated as a safety
        component and conjoined with the goal: objective = goal && safety_spec.
        If spec is None but goal exists, rolls out with just the goal.

        Args:
            spec: The specification to synthesize and run
            continue_on_timeout: If True, when rollout times out, continue with
                random walk from the final state until win or lose

        Returns:
            True if positive trace, False otherwise
        """
        if not spec and not self.goal:
            print("No spec provided for rollout, performing random rollout.")
            return self._random_rollout(continue_on_timeout=continue_on_timeout)

        # Build the objective: conjoin goal with safety spec if both exist
        if self.goal and spec:
            objective = And(self.goal, spec)
        elif self.goal:
            objective = self.goal
        else:
            objective = spec

        print(f"Rolling out with spec: {objective}")

        result = self.api.synthesize_and_run(objective=str(objective), params=self.params, timeout_steps=20)

        if result.error_message:
            print(f"Synthesis/run error: {result.error_message}")

        # Check if we should continue timed-out traces with random walk
        if continue_on_timeout and self._is_timeout_failure(result.error_message) and result.trace:
            print("Timeout detected, continuing with random rollout...")
            return self._continue_with_random(result.trace)

        return self._store_trace(result.trace, result.success, self.params,
                                 failure_reason=result.error_message)

    def loop(self, 
             iters: int = 10, 
             max_size: int = 9, 
             start_size: int = 6,
             pop_rank: str | None = "pred", 
             continue_on_timeout: bool = True):
        """
        Main RL loop: rollout, refine, repeat until spec queue exhausted.

        Args:
            iters: Maximum number of iterations.
            max_depth: Maximum depth for mining (unused currently).
            start_size: Starting formula size for collect-all mining.
            pop_rank: Ranking strategy for spec selection. "pred" prioritizes specs with
                      most predicates and fewest updates. None uses LIFO.
            continue_on_timeout: If True, when spec rollout times out, continue with
                random walk from final state until win or lose. This ensures every
                rollout produces a positive or negative trace.
        """
        # Initial rollout: use goal spec if available, otherwise random
        if self.goal:
            print(f"Initial rollout with goal spec: {self.goal}")
            self.rollout(spec=None, continue_on_timeout=continue_on_timeout)
        else:
            self._random_rollout(continue_on_timeout=continue_on_timeout)
        iteration = 0
        success = False
        size = start_size
        while iteration < iters:
            iteration += 1
            print(f"\n=== Iteration {iteration}: Spec stack size {len(self.specs.stack)} ===")
            if not success or self.spec is None:
                # Invalidate stale stack when new traces have arrived since last collect-all
                current_trace_count = len(self.pos_traces) + len(self.neg_traces)
                if current_trace_count > self._traces_at_last_collect:
                    print(f"  New traces since last collect-all ({self._traces_at_last_collect} → {current_trace_count}), clearing stale stack ({len(self.specs.stack)} specs)")
                    self.specs.stack.clear()
                    size = start_size

                self.refine_policy(opt="--first-all")  # Refine policy with new traces, get new spec(s)
                # Try collecting all alternatives if first-all didn't yield new specifications to try
                while len(self.specs.stack) == 0 and size < max_size:
                    self.refine_policy(opt="--collect-all", max_size=size)
                    self._traces_at_last_collect = len(self.pos_traces) + len(self.neg_traces)
                    size += 1
            # Get next spec from queue using ranking strategy
                self.spec = self.specs.pop(rank=pop_rank)

            print(f"Current spec: {"NONE" if not self.spec else self.spec}")

            # Rollout with current spec
            success = self.rollout(self.spec, continue_on_timeout=continue_on_timeout)
            print(f"Rollout result: {'Positive' if success else 'Negative'}")


            # print("\n=== Positive Traces ===")
            for i, trace in enumerate(self.pos_traces):
                print(f"\n--- Positive Trace {i+1} ({len(trace)} steps) ---")
                for step in trace:
                    print(f"  {step}")

            # print("\n=== Negative Traces ===")
            for i, trace in enumerate(self.neg_traces):
                print(f"\n--- Negative Trace {i+1} ({len(trace)} steps) ---")
                for step in trace:
                    print(f"  {step}")

            if success:
                print("*** Positive trace obtained! ***")
                

            # if self.spec_queue == []:
            #     print("Spec queue exhausted, no more specs to try.")
            #     print("Refine policy with new traces")
            #     # Refine policy with new trace
            # if not success:
            #     print("Spec unsuccessful, refining specification.")
            #     print("Refine policy with new traces")
            #     self.refine_policy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("game", default="frozen_lake")
    parser.add_argument("--varied", action="store_true", help="Use varied board configs")
    parser.add_argument("--work-dir", type=Path, required=False, help="Working directory for traces and outputs")
    parser.add_argument("--continue-on-timeout", action="store_true",
                        help="Continue timed-out traces with random rollout until win/lose")
    parser.add_argument("--pop-rank", choices=["pred", None], default="pred",
                        help="Ranking strategy for spec selection (default: pred)")
    parser.add_argument("--goal-specified", action="store_true",
                        help="Use the known goal/liveness spec for the game (e.g. F(reach goal))")
    args = parser.parse_args()

    goal = get_goal_spec(args.game) if args.goal_specified else None
    if args.goal_specified and goal is None:
        print(f"Warning: no known goal spec for game '{args.game}', running without goal.")

    loop = RLLoop(game=args.game, varied=args.varied, work_dir=args.work_dir, goal=goal)
    print(f"Work dir: {loop.work_dir}")
    print(f"Goal: {loop.goal}")
    print(f"Pos: {len(loop.pos_traces)}, Neg: {len(loop.neg_traces)}")
    print(f"Spec: {loop.spec}")
    # exit(0)
    loop.loop(pop_rank=args.pop_rank, continue_on_timeout=args.continue_on_timeout)
    print("\n=== Final Results ===")
    print(f"Final spec: {loop.spec}")
    print(f"Total positive traces: {len(loop.pos_traces)}")
    print(f"Total negative traces: {len(loop.neg_traces)}")

    if loop.pos_traces:
        print("\n=== Positive Traces ===")
        for i, trace in enumerate(loop.pos_traces):
            print(f"\n--- Positive Trace {i+1} ({len(trace)} steps) ---")
            for step in trace:
                print(f"  {step}")

        print("\n=== Negative Traces ===")
        for i, trace in enumerate(loop.neg_traces):
            print(f"\n--- Negative Trace {i+1} ({len(trace)} steps) ---")
            for step in trace:
                print(f"  {step}")
    



