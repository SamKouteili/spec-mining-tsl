import pydot
import random
import argparse
from pathlib import Path

from utils import *
from utilsf import get_input
from runner import run_dot_gen

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hoa", required=False, help="Input HOA file to convert to dot")
    parser.add_argument("--check", type=str, required=False, help="Check if spot word is accepted by input hoa")
    parser.add_argument("--gen", type=int, required=False, help="Number of traces to generate")
    parser.add_argument("--length", type=int, default=6, help="Length of traces to generate")
    return parser.parse_args()

class Dotomata:    
    """A simple parser for DOT format to extract states and transitions."""

    @staticmethod
    def load_dot(dot: str):
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
        initial = init_edges[0].get_destination().strip('"') if init_edges else states[0]

        transitions = {}
        accepting_transitions = set()  # Track accepting transitions for trans-acc
        alphabet = set()

        for e in graph.get_edges():
            src = e.get_source().strip('"')
            dst = e.get_destination().strip('"')
            if src == "I":
                continue
            label = e.get_label().strip('"')

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
            alphabet.add(inp)
            transitions.setdefault(src, {})[inp] = (dst, out)

            if is_accepting_trans:
                accepting_transitions.add((src, inp, dst))

        return {
            "states": states,
            "initial": initial,
            "alphabet": sorted(alphabet),
            "transitions": transitions,
            "accepting": accepting_states,
            "accepting_transitions": accepting_transitions,
        }

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


    # --- Trace generation ---
    @staticmethod
    def ex_dot_machine(machine, length=10, cycle=True):
        state = machine["initial"]
        transitions = machine["transitions"]

        trace = []
        for _ in range(length):
            valid_inputs = list(transitions[state].keys())
            if not valid_inputs:
                break
            inp = random.choice(valid_inputs)
            next_state, out = transitions[state][inp]
            trace.append(f"{inp}/{out}")
            state = next_state

        if cycle:
            trace.append("cycle{1}")
        return ";".join(trace)

    @staticmethod
    def evaluate_formula(formula: str, letter: Letter) -> bool:
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

    @staticmethod
    def check_trace_acceptance_dot(dot_machine, trace: Trace, debug=False) -> bool:
        """Check if a trace is accepted by a DOT machine.

        Walks through the machine following the trace and checks acceptance.
        Supports both state-acc and trans-acc semantics.

        Args:
            dot_machine: dict from load_dot() with 'initial', 'transitions', 'accepting', 'accepting_transitions'
            trace: Trace object (List[List[str]]) in spot format
            debug: If True, print debug information

        Returns:
            bool: True if trace is accepted (state-acc: ends in accepting state, trans-acc: uses accepting transition)
        """
        state = dot_machine["initial"]
        transitions = dot_machine["transitions"]
        accepting_states = dot_machine.get("accepting", set())
        accepting_transitions = dot_machine.get("accepting_transitions", set())

        # Track if we've taken any accepting transition (for trans-acc)
        used_accepting_trans = False

        for i, letter in enumerate(trace):
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
    args = parser()
    args = vars(args)
    # print(args)

    assert not (args.get("check") and args.get("gen")), "Cannot check acceptance and gen"


    hoa = get_input() if not args.get("hoa") else Path(args["hoa"]).read_text()

    aps = get_ap_list(hoa)

    dot = run_dot_gen(hoa)
    machine = Dotomata.load_dot(dot)

    if args.get("check"):
        if Dotomata.check_trace_acceptance_dot(machine, str_to_trace(args["check"])) :
            print("accepted")
            exit(0)
        else :
            print("rejected")
            exit(1)
    elif args.get("gen"):
        for i in range(args["gen"]):
            out = Dotomata.ex_dot_machine(machine, args['length'])
            print(Dotomata.machine_to_spot(out, aps))
    else:
        print(dot)
     
