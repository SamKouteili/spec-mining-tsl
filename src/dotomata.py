import pydot
import random
import argparse
from pathlib import Path

# from tracer import Trace, get_ap_list
from utils import *
from runners import run_dot_gen

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=False, help="Input HOA file to convert to dot")
    parser.add_argument("--hoa", action="store_true", help="Input file format is hoa")
    parser.add_argument("--dot", action="store_true", help="Input file format is dot")
    parser.add_argument("--check", type=str, required=False, help="Check if spot word is accepted by input hoa")
    parser.add_argument("--gen", type=int, required=False, help="Number of traces to generate")
    parser.add_argument("--length", type=int, default=6, help="Length of traces to generate")
    parser.add_argument("-p", action="store_true", help="Generate positive traces")
    parser.add_argument("-n", action="store_true", help="Generate negative traces")
    parser.add_argument("-d", action="store_true", help="Debug")
    return parser.parse_args()


class Dotomata:
    """A simple parser for DOT format to extract states and transitions."""

    @staticmethod
    def load_hoa(hoa: str):
        """Parse HOA format directly without converting to DOT.

        Returns a Dotomata object.
        """
        lines = hoa.strip().splitlines()

        # Parse header
        num_states = None
        initial = None
        aps = []
        acc_name = None
        is_state_acc = False
        is_trans_acc = False

        for line in lines:
            line = line.strip()
            if line.startswith("States:"):
                num_states = int(line.split(":")[1].strip())
            elif line.startswith("Start:"):
                initial = line.split(":")[1].strip()
            elif line.startswith("AP:"):
                parts = line.split()
                # Format: AP: <count> "ap1" "ap2" ...
                ap_count = int(parts[1])
                aps = [parts[i].strip('"') for i in range(2, 2 + ap_count)]
            elif line.startswith("acc-name:"):
                acc_name = line.split(":")[1].strip()
            elif line.startswith("properties:"):
                # Multiple properties lines may exist, accumulate flags
                if "state-acc" in line:
                    is_state_acc = True
                if "trans-acc" in line:
                    is_trans_acc = True

        if num_states is None or initial is None:
            raise ValueError("HOA missing States or Start header")

        # Parse body
        states = [str(i) for i in range(num_states)]
        accepting_states = set()
        accepting_transitions = set()
        transitions = {}

        in_body = False
        current_state = None

        for line in lines:
            line = line.strip()

            if line == "--BODY--":
                in_body = True
                continue
            elif line == "--END--":
                break

            if not in_body:
                continue

            if line.startswith("State:"):
                # Format: "State: 0" or "State: 0 {0}" (with acceptance set)
                parts = line.split()
                current_state = parts[1]
                # Check for state acceptance
                if is_state_acc and "{" in line:
                    accepting_states.add(current_state)
            elif line.startswith("["):
                # Transition line: [formula] dest or [formula] dest {0}
                # Extract formula and destination
                bracket_end = line.index("]")
                formula_bits = line[1:bracket_end]
                rest = line[bracket_end + 1:].strip().split()

                dest_state = rest[0]
                is_acc_trans = len(rest) > 1 and "{0}" in rest[1]

                # Convert bit formula to spot formula
                formula = Dotomata.bits_to_formula(formula_bits, aps)

                # Normalize formula before storing
                normalized_formula = Dotomata.normalize_formula(formula)

                # Store transition
                transitions.setdefault(current_state, {})[normalized_formula] = (dest_state, "")

                if is_trans_acc and is_acc_trans:
                    accepting_transitions.add((current_state, normalized_formula, dest_state))

        # Create a Dotomata object from parsed data
        machine = Dotomata.__new__(Dotomata)
        machine.states = states
        machine.initial = initial
        # Normalize formulas before storing in alphabet
        machine.alphabet = sorted(set(
            Dotomata.normalize_formula(formula)
            for trans in transitions.values()
            for formula in trans.keys()
        ))
        machine.transitions = transitions
        machine.accepting = accepting_states
        machine.accepting_transitions = accepting_transitions
        return machine

    @staticmethod
    def normalize_formula(formula: str) -> str:
        """Normalize a boolean formula to canonical form for comparison.

        Args:
            formula: Boolean formula like "p0 & p1" or "!p0 | p1"

        Returns:
            Normalized formula with sorted terms
        """
        formula = formula.strip()

        # Handle special cases
        if formula in ("1", "t", ""):
            return "1"
        if formula in ("0", "f"):
            return "0"

        # Handle disjunctions (OR)
        if "|" in formula:
            # Split by | and normalize each disjunct
            disjuncts = []
            depth = 0
            current = []
            for char in formula:
                if char == '(':
                    depth += 1
                    current.append(char)
                elif char == ')':
                    depth -= 1
                    current.append(char)
                elif char == '|' and depth == 0:
                    disjuncts.append(''.join(current).strip())
                    current = []
                else:
                    current.append(char)
            if current:
                disjuncts.append(''.join(current).strip())

            # Normalize each disjunct and sort
            normalized_disjuncts = sorted([Dotomata.normalize_formula(d.strip("() ")) for d in disjuncts])
            if len(normalized_disjuncts) == 1:
                return normalized_disjuncts[0]
            return "(" + " | ".join(normalized_disjuncts) + ")"

        # Handle conjunctions (AND)
        # Parse into individual literals
        tokens = [tok.strip() for tok in re.split(r"&", formula) if tok.strip()]

        # Sort the literals for canonical form
        tokens = sorted(tokens)

        if not tokens:
            return "1"

        return " & ".join(tokens)

    @staticmethod
    def bits_to_formula(bits: str, aps: list[str]) -> str:
        """Convert HOA bit formula to spot-style formula.

        Args:
            bits: e.g., "0&!1&2" or "t" (true/tautology)
            aps: list of atomic proposition names

        Returns:
            spot formula like "p0 & !p1 & p2"
        """
        bits = bits.strip()

        # Handle special cases
        if bits == "t" or bits == "":
            return "1"  # true
        if bits == "f":
            return "0"  # false

        # Parse the bit formula
        # Split by | for disjunctions
        if "|" in bits:
            disjuncts = [d.strip() for d in bits.split("|")]
            converted = [Dotomata.bits_to_formula(d, aps) for d in disjuncts]
            return "(" + " | ".join(converted) + ")"

        # Parse conjunctions
        conjuncts = [c.strip() for c in bits.split("&")]
        result = []

        for conj in conjuncts:
            if not conj:
                continue

            # Check for negation
            if conj.startswith("!"):
                ap_idx = int(conj[1:])
                result.append(f"!{aps[ap_idx]}")
            else:
                ap_idx = int(conj)
                result.append(aps[ap_idx])

        if not result:
            return "1"

        return " & ".join(result)

    def __init__(self, dot: str):
        graphs = pydot.graph_from_dot_data(dot)
        graph = graphs[0] if graphs else None
        if graph is None:
            raise ValueError("No graph found in DOT data")

        states = []
        accepting_states = set()

        for n in graph.get_nodes():
            name = n.get_name().strip('"')
            if name not in ("node", "I"):
                states.append(name)
                # Check if accepting (peripheries=2 means double circle in DOT)
                if n.get_attributes().get('peripheries') == '2':
                    accepting_states.add(name)

        # find initial state via I -> X edge
        init_edges = [e for e in graph.get_edges() if e.get_source() == "I"]
        initial = init_edges[0].get_destination().strip('"') if init_edges else states[0] # type: ignore

        transitions = {}
        accepting_transitions = set()  # Track accepting transitions for trans-acc
        alphabet = set()

        for e in graph.get_edges():
            src = e.get_source().strip('"') # type: ignore
            dst = e.get_destination().strip('"') # type: ignore
            if src == "I":
                continue
            label = e.get_label().strip('"') # type: ignore

            # Check for trans-acc marking in label (e.g., "formula\n{0}")
            is_accepting_trans = False
            if "\\n{0}" in label or "\n{0}" in label:
                is_accepting_trans = True
                # Remove the {0} marking from the label
                label = label.replace("\\n{0}", "").replace("\n{0}", "").strip()

            if "/" in label:
                inp, out = label.split("/")
                inp, out = inp.strip(), out.strip()
            else:
                inp, out = label.strip(), ""

            # Normalize formula before storing
            normalized_inp = Dotomata.normalize_formula(inp)

            alphabet.add(normalized_inp)
            transitions.setdefault(src, {})[normalized_inp] = (dst, out)

            if is_accepting_trans:
                accepting_transitions.add((src, normalized_inp, dst))

        self.states = states
        self.initial = initial
        # Alphabet already contains normalized formulas
        self.alphabet = sorted(alphabet)
        self.transitions = transitions
        self.accepting = accepting_states
        self.accepting_transitions = accepting_transitions
    
    def __str__(self):
        return "\n\t".join(["dotomata machine:",
             f"states:{self.states}",
             f"alphabet:{self.alphabet}",
             f"initial:{self.initial}",
             f"accepting:{self.accepting}",
             f"transistions:{self.transitions}",
        ])

    # --- Spot conversion helpers ---
    @staticmethod
    def parse_formula_side(side):
        literals = {}
        if not side:
            return literals
        tokens = [tok.strip() for tok in re.split(r"&", side) if tok.strip()]
        for tok in tokens:
            if tok.startswith("!"):
                literals[tok[1:].strip()] = 0
            else:
                literals[tok.strip()] = 1
        return literals

    @staticmethod
    def simplify_disjunction(expr):
        if "|" in expr:
            return expr.split("|", 1)[0].strip("() ")
        return expr.strip("() ")

    @staticmethod
    def step_to_spot(step, ap_order):
        if "/" in step:
            inp, out = step.split("/", 1)
        else:
            inp, out = step, ""
        inp, out = inp.strip(), out.strip()

        inp = Dotomata.simplify_disjunction(inp)
        out = Dotomata.simplify_disjunction(out)

        literals = {}
        literals.update(Dotomata.parse_formula_side(inp))
        literals.update(Dotomata.parse_formula_side(out))

        # now build full AP valuation
        bits = []
        for ap in ap_order:
            if ap in literals:
                val = literals[ap]
            else:
                val = random.choice([0, 1])  # randomize unmentioned APs
            bits.append(ap if val else f"!{ap}")
        return "&".join(bits)

    @staticmethod
    def machine_to_spot(trace, ap_order):
        steps = []
        cycle_part = ""
        if "cycle{" in trace:
            prefix, cycle_part = trace.split("cycle{", 1)
            cycle_part = "cycle{" + cycle_part
            raw_steps = [s for s in prefix.split(";") if s.strip()]
        else:
            raw_steps = [s for s in trace.split(";") if s.strip()]
        for st in raw_steps:
            steps.append(Dotomata.step_to_spot(st, ap_order))
        if cycle_part:
            steps.append(cycle_part)
        return ";".join(steps)


    # --- Trace generation ---
    def execute(self, length=10, cycle=True, mode="random"):
        """Execute random walk on automaton.

        Args:
            length: Number of steps to execute
            cycle: Whether to append cycle{1} at the end
            mode: Execution mode - one of:
                - "random": Use all transitions (uniform random walk)
                - "positive": Use only accepting transitions
                - "negative": Use only non-accepting transitions

        Returns:
            String representation of execution trace, or None if no valid path exists
        """
        state = self.initial
        transitions = self.transitions
        accepting_set = self.accepting_transitions

        trace = []
        for _ in range(length):
            if state not in transitions:
                return None

            # Get valid inputs based on mode
            all_inputs = list(transitions[state].keys())

            if mode == "random":
                valid_inputs = all_inputs
            elif mode == "positive":
                # Filter to only accepting transitions
                valid_inputs = [inp for inp in all_inputs
                               if (state, inp, transitions[state][inp][0]) in accepting_set]
            elif mode == "negative":
                # Filter to only non-accepting transitions
                valid_inputs = [inp for inp in all_inputs
                               if (state, inp, transitions[state][inp][0]) not in accepting_set]
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'random', 'positive', or 'negative'")

            if not valid_inputs:
                return None

            inp = random.choice(valid_inputs)
            next_state, out = transitions[state][inp]
            trace.append(f"{inp}/{out}")
            state = next_state

        if cycle:
            trace.append("cycle{1}")
        return ";".join(trace)

    @staticmethod
    def evaluate_formula(formula: str, letter: list[str]) -> bool:
        """Evaluate a boolean formula given a letter (truth assignment).

        Args:
            formula: Boolean formula like "p0 & p1" or "!p0 | p1" or "1"
            letter: Spot letter like ["p0", "!p1", "p2"]

        Returns:
            True if the letter satisfies the formula
        """
        # Build truth assignment from letter
        true_aps = {ap for ap in letter if not ap.startswith("!")}
        false_aps = {ap[1:] for ap in letter if ap.startswith("!")}

        # Handle special cases
        formula = formula.strip()
        if formula in ("1", "t", ""):
            return True

        # Handle disjunctions: formula is a disjunction of conjunctions
        # Split by | to get disjuncts
        if "|" in formula:
            disjuncts = [d.strip("() ") for d in re.split(r"\|", formula)]
            # At least one disjunct must be satisfied
            for disjunct in disjuncts:
                if Dotomata.evaluate_conjunction(disjunct, true_aps, false_aps):
                    return True
            return False
        else:
            # Single conjunction
            return Dotomata.evaluate_conjunction(formula, true_aps, false_aps)

    @staticmethod
    def evaluate_conjunction(conj: str, true_aps: set, false_aps: set) -> bool:
        """Evaluate a conjunction of literals."""
        conj = conj.strip()
        if conj in ("1", "t", ""):
            return True

        # Parse conjunctions
        tokens = [tok.strip() for tok in re.split(r"&", conj) if tok.strip()]

        for tok in tokens:
            tok = tok.strip("() ")
            if tok.startswith("!"):
                # Negative literal
                ap = tok[1:]
                if ap in true_aps:
                    return False
            else:
                # Positive literal
                if tok in false_aps:
                    return False

        return True

    # --- Trace Acceptance ---
    def accepts(self, trace, debug=False) -> bool:
        """Check if a trace is accepted by a DOT machine.
            Walks through the machine following the trace and checks acceptance.
            Supports both state-acc and trans-acc semantics.

            @param trace: Trace object (List[List[str]]) in spot format
            @returns bool: True if trace is accepted 
                (state-acc: ends in accepting state, trans-acc: uses accepting transition)
        """
        trace_ = trace.trace()
        state = self.initial
        transitions = self.transitions
        accepting_states = self.accepting
        accepting_transitions = self.accepting_transitions

        # Track if we've taken any accepting transition (for trans-acc)
        used_accepting_trans = False

        for i, letter in enumerate(trace_):
            if state not in transitions:
                if debug:
                    print(f"[check] Step {i}: state {state} has no transitions")
                return False

            # Find a transition whose input formula is satisfied by this letter
            found = False
            for inp, (next_state, _) in transitions[state].items():
                if Dotomata.evaluate_formula(inp, letter):
                    # Check if this transition is accepting (for trans-acc)
                    if (state, inp, next_state) in accepting_transitions:
                        used_accepting_trans = True
                        if debug:
                            print(f"[check] Step {i}: {state} --[{inp}]--> {next_state} (ACCEPTING)")
                    else:
                        if debug:
                            print(f"[check] Step {i}: {state} --[{inp}]--> {next_state}")
                    state = next_state
                    found = True
                    break

            if not found:
                if debug:
                    print(f"[check] Step {i}: No transition from {state} matches letter {letter}")
                    print(f"[check]   Available transitions: {list(transitions[state].keys())}")
                return False  # No valid transition for this letter

        # Check acceptance: state-acc OR trans-acc
        state_acc = state in accepting_states
        trans_acc = used_accepting_trans and len(accepting_transitions) > 0

        result = state_acc or trans_acc
        if debug:
            print(f"[check] Final state: {state}, state-acc: {state_acc}, trans-acc: {trans_acc}, result: {result}")
        return result
    

if __name__ == "__main__" :
    from tracer import Trace
    import difflib
    args = parser()
    args = vars(args)
    # print(args)

    assert not (args.get("check") and args.get("gen")), "Cannot check acceptance and gen"
    assert not (args.get("p") and args.get("n")), "Cannot generate positive and negative traces"

    mode = "positive" if args.get("p") else "negative" if args.get("n") else "random"
    if args.get("d"):
        print(f"Trace generation type: {mode}")


    hoa = get_input() if not args.get("i") else Path(args["i"]).read_text()
    aps = get_ap_list(hoa)

    machine = Dotomata.load_hoa(hoa)

    # if args.get("d"):
    #     m2 = Dotomata(run_dot_gen(hoa))

    #     s1 = str(m2)
    #     s2 = str(machine)
    #     diff_lines = list(difflib.unified_diff(s1.splitlines(), s2.splitlines(), fromfile='m2', tofile='machine', lineterm=''))
    #     if diff_lines:
    #         print("\n".join(diff_lines))
    #     else:
    #         print("No differences between m2 and machine")

        # assert str(m2) == str(machine), "machines not equivalent"

    if args.get("check"):
        trace = Trace.from_spot(args["check"]).change_alphabet(aps)
        if machine.accepts(trace, debug=bool(args.get('d'))) :
            print("accepted")
            exit(0)
        else :
            print("rejected")
            exit(1)
    elif args.get("gen"):
        for i in range(args["gen"]):
            out = machine.execute(args['length'], mode=mode)
            print(Dotomata.machine_to_spot(out, aps))
    else:
        print(machine)
     


    # @staticmethod
    # def convert_machine_trace_to_sub_alphabet(machine_trace: str, new_aps: list[str]) -> str:
    #     """Convert a machine trace to only mention APs in new_aps.

    #     Args:
    #         machine_trace: trace in machine format (e.g., "p0 & p1/;p2/;cycle{1}")
    #         new_aps: target atomic propositions to keep

    #     Returns:
    #         trace string in machine format with only new_aps mentioned
    #     """
    #     # Handle cycle notation
    #     cycle_part = ""
    #     if "cycle{" in machine_trace:
    #         prefix, cycle_part = machine_trace.split("cycle{", 1)
    #         cycle_part = ";cycle{" + cycle_part
    #         steps = [s for s in prefix.split(";") if s.strip()]
    #     else:
    #         steps = [s for s in machine_trace.split(";") if s.strip()]

    #     new_steps = []
    #     for step in steps:
    #         # Split input/output
    #         if "/" in step:
    #             inp, out = step.split("/", 1)
    #         else:
    #             inp, out = step, ""

    #         # Parse and filter input formula (conjunction of literals)
    #         inp = inp.strip()
    #         if inp in ("1", "t", ""):  # true or empty
    #             filtered_literals = []
    #         else:
    #             tokens = [tok.strip() for tok in re.split(r"&", inp) if tok.strip()]
    #             filtered_literals = []
    #             for tok in tokens:
    #                 ap_name = tok.lstrip("!")
    #                 if ap_name in new_aps:
    #                     filtered_literals.append(tok)

    #         # Reconstruct step
    #         new_inp = " & ".join(filtered_literals) if filtered_literals else "1"
    #         new_steps.append(f"{new_inp}/{out}" if out else new_inp)

    #     return ";".join(new_steps) + cycle_part