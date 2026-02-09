"""Generating finite-prefix accepting traces"""
from __future__ import annotations
import argparse
import json
import time
import random
from pathlib import Path
# from typing import Dict

from runners import *
from utils import *


def parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("context", type=str, 
    #                     help="Context: mine or eval")
    parser.add_argument("--tsl", type=str, required=True,
                        help="Input TSL File (for Atomic Propositions)")
    # parser.add_argument("--tslm", type=str, required=False,
    #                     help="Input Mined TSL File (only for eval context)")
    # parser.add_argument("--ltlm", type=str, required=False)
    # parser.add_argument("--hoax", type=str, required=True,
    #                     help="Input Hoax Execution Directory")
    parser.add_argument("-o", type=str,
                        help="Output trace file")
    parser.add_argument("-t", type=str, default="spot", choices=["scarlet", "spot", "bolt"],
                        help="Type of trace to generate")
    parser.add_argument("-p", type=int, default=10,
                        help="Number of positive traces to generate")
    parser.add_argument("-n", type=int, default=10,
                        help="Number of negative traces to generate")
    parser.add_argument("-l", type=int, default=5,
                        help="Generated trace length")
    parser.add_argument("--timeout", type=float, default=5,
                        help="Trace generation timeout (in seconds)")
    return parser.parse_args()


class Trace:
    # initialise from spot trace
    def __init__(self, trace: list[list[str]], aps = None):
        # self._trace: list[list[str]]
        # self._aps: list[str]
        # split_var = "&" if kind == "spot" else ","
        self._trace = trace
        self._aps = aps if aps else list(dict.fromkeys(ap.strip("!") for ap in self._trace[0]))
    
    def __eq__(self, other):
        return self.trace() == other.trace()
    
    def __str__(self):
        return str(self._trace)
    
    def trace(self) -> list[list[str]]:
        return self._trace
    
    def aps(self) -> list[str]:
        return self._aps
    
    @staticmethod
    def spot2scarlet(spot_trace: str) -> str:
        """Convert a spot trace to scarlet trace."""
        return ";".join([",".join(["0" if ap.strip().startswith("!") else "1" for ap in letter.split("&")])
                         for letter in spot_trace.strip("cycle{1}").strip().split(";")])

    @staticmethod
    def scarlet2spot(scarlet_trace: str, aps: list[str]) -> str:
        """Convert a scarlet trace to a spot trace"""
        return ";".join([" & ".join([aps[i] if int(ap) == 1 else f"!{aps[i]}" for i, ap in enumerate(letter.strip().split(","))])
                         for letter in scarlet_trace.strip().split(",")])

    @classmethod
    def from_spot(cls, spot_trace: str) -> Trace:
        t = [[ap.strip() for ap in step.strip().split("&") if ap.strip()] 
                for step in spot_trace.split(";") if step.strip() and not step.startswith("cycle{")]
        return Trace(t)
    
    @classmethod
    def from_scarlet(cls, scarlet_trace: str, aps: list[str]) -> Trace:
        return Trace.from_spot(Trace.scarlet2spot(scarlet_trace, aps))
    
    @classmethod
    def from_hoax(cls, hoax_trace: str, aps: list[str]) -> Trace:
        """Generate trace list from hoax output (Spot/Scarlet)."""
        trace = []
        for line in hoax_trace.splitlines():
            raw = re.search(r"{(.*)}", line)
            if not raw:
                continue
            present = [tok.strip("'\" ")
                    for tok in raw.group(1).split(",") if tok.strip()]
            
            assignment = [ap if ap in present else f"!{ap}" for ap in aps]
            trace.append(assignment)
        return Trace(trace)

    def to_spot(self) -> str:
        if not all(self._trace):
            return "cycle{1}"
        return ";".join([" & ".join(letter) for letter in self._trace]) + ";cycle{1}"
    
    def to_scarlet(self) -> str:
        return ";".join([",".join(["0" if ap.startswith("!") else "1" for ap in letter]) for letter in self._trace])
    
    def to_bolt(self) -> dict[str, list[int]] :
        d = {}
        for e in self._trace[0]:
            e = e.strip("!").strip()
            for letter in self._trace:
                d[e] = d.get(e, []) + ([0] if e not in letter else [1])
        return d


    # might be buggy
    def change_alphabet(self, new_aps: list[str]) -> Trace:
        """Convert a trace to a new alphabet of atomic propositions.
            Assume new alphabet is a subset or superset of original alphabet."""
        if len(new_aps) < len(self._aps):
            return Trace([[ap for ap in letter if ap.strip("!") in new_aps] for letter in self._trace], new_aps)
        elif len(new_aps) > len(self._aps):
            # dif_aps = list(set(new_aps).difference(set(self._aps)))
            dif_aps = [ap for ap in new_aps if ap not in self._aps]
            return Trace([word + [ap if random.choice([True, False]) else f"!{ap}" for ap in dif_aps] for word in self._trace], new_aps)
        else:
            return self

    def update_mutex_consistent(self) -> bool:
        """Check if a spot trace is valid (mutual exclusion of updates preserved).
            Suppose trace is in spot format."""

        # find all update terms
        update_terms = set()
        for ap in self._trace[0]:
            if ap.strip("!").startswith("u0"):
                base_ap = ap.strip("!").strip("u0").split("0")[0]
                update_terms.add(base_ap)
        
        for letter in self._trace:
            update_map = set()
            for ap in letter:
                if ap.startswith("u0") : # postivie update variable (so we are updating here)
                    base_ap = ap.strip("u0").split("0")[0]
                    if base_ap in update_map: # if updating the same variable again in the same letter
                        # Mutual exclusion conflict: Cannot update the same variable twice at same timestep 
                        # print("Conflict on", letter)
                        return False
                    update_map.add(base_ap)
            if update_map != update_terms :
                # Did not update the variable at least once
                return False
        # print("Valid trace:", spot_trace)
        return True




def get_assumption_hoa(f: str, delim="ASSUME") -> tuple[str, list[str]] :
    ltla = " & ".join([l.strip("\n") for l in extract_block(f, delim).split(";") if l.strip("\n")]).replace(
        "&&", "&").replace("||", "|").strip()
    
    if ltla == "":
        print("[tracer] Generating traces... no assumptions found. Supposing True.")
        hoaa = run_ltl2tgba("True")
        apsa = []
    else :
        # Use ltl2tgba for assumptions (not ltlf2dfa) because assumptions often contain
        # temporal operators like X, W, R that are not supported in LTLf
        hoaa = run_ltl2tgba(ltla, flags=[]) # flags=["-M", "-D", "-H"]
        # print("AssUMPTION HOA:\n", hoaa)
        apsa = get_ap_list(hoaa)

    return hoaa, apsa




def generate_traces(num, length, positive, tsl, timeout=10) -> tuple[list[Trace],list[str]]:
    """Generate traces for some ltl specification that adhere to some assumption ltl.
        Positive traces are accepted by both the hoa and the assumption hoa.
        return: list[traces], aps """
    from dotomata import Dotomata
    
    tlsf = run_tsl("tlsf", tsl)
    ltl = run_syfco(tlsf, flags=["-f", "ltl"])
    # ltlf = run_ltlf(ltl)
    # hoa = run_to_finite(run_ltl2tgba(ltlf, flags=["-B"]))
    hoa = run_ltlf2dfa(ltl)
    aps = get_ap_list(hoa)
    # dot = run_dot_gen(hoa)
    machine = Dotomata.load_hoa(hoa)

    print("[tracer] TSL:\n",tsl)
    print("[tracer] LTL:\n",ltl)
    print("[tracer] HOA:\n",hoa)
    print("[tracer] APs:", aps)
    print("[tracer] Machine\n:", machine)

    hoaa, apsa = get_assumption_hoa(tlsf)
    print("[tracer] AssumptionHOA:\n", hoaa)
    print("[tracer] AssumptionAPs:", apsa)


    traces: list[Trace] = []
    start_time = time.time()
    mode = "random" if positive else "negative" # NOTE: change random to positive?
    # m = machine
    while len(traces) < num and (time.time() - start_time < timeout)  :

        exec_out = machine.execute(length=length, cycle=True, mode=mode)
        if exec_out is None:
            print("exec_out is None")
            continue

        spot_str = Dotomata.machine_to_spot(exec_out, aps)
        trace = Trace.from_spot(spot_str)
    
        if trace not in traces and trace.update_mutex_consistent():

            trace_assumption = trace.change_alphabet(apsa)
            # print(trace_assumption)
            if run_accept_word(hoaa, trace_assumption.to_spot()): 

                if positive and machine.accepts(trace, debug=False) :
                        traces.append(trace)
                
                if not positive and not machine.accepts(trace, debug=False):
                        traces.append(trace)

    
    print(f"[tracer] Generated {len(traces)} {positive} traces")

    return traces, aps
    
    

def generate_traces_with_neg(num, length, positive, tsl, timeout=10) -> tuple[list[Trace],list[str]]:
    """Generate traces for some ltl specification that adhere to some assumption ltl.
        Positive traces are accepted by both the hoa and the assumption hoa.
        return: list[traces], aps """
    from dotomata import Dotomata

    tlsf = run_tsl("tlsf", tsl)
    ltl = run_syfco(tlsf, flags=["-f", "ltl"])
    # ltlf = run_ltlf(ltl)
    # hoa = run_to_finite(run_ltl2tgba(ltlf, flags=["-B"]))
    hoa = run_ltlf2dfa(ltl)
    aps = get_ap_list(hoa)
    # dot = run_dot_gen(hoa)
    machine = Dotomata.load_hoa(hoa)

    # # For negative traces, also keep the positive DFA for checking
    # nmachine, naps = Dotomata(""), []
    if not positive:
        ntlsf = negate_tlsf(tlsf)
        nltl = run_syfco(ntlsf, flags=["-f", "ltl"])
        nhoa = run_ltlf2dfa(nltl)
        naps = get_ap_list(nhoa)
        # ndot = run_dot_gen(nhoa)
        nmachine = Dotomata.load_hoa(nhoa)
        # nmachine = Dotomata(ndot)
        # print("[tracer] nTSL:\n", ntsl)
        print("[tracer] nLTL:\n", nltl)
        print("[tracer] nHOA:\n", nhoa)
        print("[tracer] nAPs:", naps)
        print("[tracer] nMachine:", nmachine)


    hoaa, apsa = get_assumption_hoa(tlsf)

    traces: list[Trace] = []
    start_time = time.time()
    m = machine if positive else nmachine # type: ignore
    # m = machine
    while len(traces) < num and (time.time() - start_time < timeout)  :

        # Use mode="random" for positive traces, mode="negative" for negative traces
        # mode = "random" if positive else "negative"
        exec_out = m.execute(length=length, cycle=True, mode="random")

        if exec_out is None:
            continue

        # print(exec_out)
        spot_str = Dotomata.machine_to_spot(exec_out, aps)
        trace = Trace.from_spot(spot_str)

        # Consider two sources of negative traces: one from random walk of original automata
        #  and one of the complement tsl automata
        # if not positive:
        #     ntrace_orig = Dotomata.ex_dot_machine(nmachine, length=length, cycle=True)
        #     ntrace_str = Dotomata.machine_to_spot(ntrace_orig, aps)
        #     ntrace_var = str_to_trace(ntrace_str)
        #     if len(aps) != len(naps) : # negation deleted some variable
        #             dif_aps = list(set(naps).difference(set(aps)))
        #             print("dif_aps:", dif_aps)
        #             ntrace_var = add_aps_to_trace(ntrace_var, dif_aps)
        #     trace_vars.append(ntrace_var)

        if trace not in traces and trace.update_mutex_consistent():

            trace_assumption = trace.change_alphabet(apsa)
            # print(trace_assumption)
            if run_accept_word(hoaa, trace_assumption.to_spot()): 
                # and Dotomata.check_trace_acceptance_dot(dota_exec, trace_vars_a): # NOTE: dot or hoa check?
                # and run_accept_word(hoaa, trace_to_str(trace_vars_a, "spot")): 
                # print(f"[tracer] {positive} candidate: {trace_str}")
                # Check acceptance in main automaton
                # if positive or not run_accept_word(hoafp, trace_str) :

                # print(f"[tracer] candidate: {spot_str}")
                # print(spot_str)
                # print(trace_assumption.to_spot())

                if positive :
                    if machine.accepts(trace, debug=False) :
                        traces.append(trace)
                else:
                    trace = trace.change_alphabet(aps) # in case negation alphabet is smaller

                    # print("\t accept:", Dotomata.check_trace_acceptance_dot(machine, trace_var, debug=True))
                    # print("\tlen trave_var_spot:", len(traces_vars_spot))
                    if not machine.accepts(trace, debug=False):
                        traces.append(trace)


    print(f"[tracer] Generated {len(traces)} {positive} traces")

    return traces, aps

def print_traces(traces: list[Trace]):
    for t in traces:
        for l in t.trace():
            print("\t", l)
        print(t.to_spot())
        print("-----")

def write_trace_file(pos_traces: list[Trace], 
                     neg_traces: list[Trace], 
                     aps: list[str],
                     kind: str,
                     out_path: str | None):
    
    pos_traces_str, neg_traces_str = [[t.to_spot() if kind == "spot" else
                                       t.to_scarlet() if kind == "scarlet" else 
                                       t.to_bolt() for t in traces]
                                      for traces in [pos_traces, neg_traces]]
    
    if kind == "bolt":
        d = {
            "positive_traces": pos_traces_str,
            "negative_traces": neg_traces_str,
            "atomic_propositions": aps,
            "number_atomic_propositions": len(aps),
            "number_traces": len(pos_traces_str) + len(neg_traces_str), # type: ignore
            "number_positive_traces": len(pos_traces_str),
            "number_negative_traces": len(neg_traces_str),
            "max_length_traces": max(len(t.trace()) for t in pos_traces + neg_traces), # type: ignore
            "trace_type:": "finite"
        }
        s = json.dumps(d)

    else: 
        s = ""
        for trace in pos_traces_str:
            s += trace + "\n" # type: ignore 
        if len(neg_traces_str) > 0 or kind == "scarlet":
            s += "---\n"
        for trace in neg_traces_str:
            s += trace + "\n" # type: ignore

        if kind == "scarlet":
            s += "---\n"
            s += "F,G,X,!,&,|\n"
            s += "---\n"
            if pos_traces:
                s += ",".join(aps) + "\n"
    
    if out_path:
        with open(out_path, "w") as f:
            f.write(s)
    else:
        print(s)

def main(args):

    tsl = Path(args["tsl"]).read_text()

    pos_traces, aps = generate_traces(tsl=tsl,num=args['p'],
                                            length=args['l'],
                                            positive=True,
                                            timeout=args["timeout"])
    assert args["p"] == 0 or pos_traces, "Unable to generate any positive traces"
    # print("[tracer] positive traces generated")

    neg_traces, _ = generate_traces(tsl=tsl,num=args['n'],
                                          length=args['l'],
                                          positive=False,
                                          timeout=args["timeout"])
    
    if len(neg_traces) < args["n"]:
        print(f"[tracer] Unable to generate {args["n"]} negative traces (generated {len(neg_traces)}). Calling gen_traces_neg...")
        neg_traces_2, _ = generate_traces_with_neg(tsl=tsl, num=args['n'] - len(neg_traces),
                                                   length=args['l'],
                                                   positive=False,
                                                   timeout=args["timeout"])
        neg_traces += neg_traces_2

    assert args["n"] == 0 or neg_traces, "Unable to generate any negative traces"

    print("* pos_traces:")
    print_traces(pos_traces)
    print("* neg_traces_vars")
    print_traces(neg_traces)

    print(f"[tracer] generated {len(pos_traces)} positive | {len(neg_traces)} negative traces")

    write_trace_file(pos_traces=pos_traces, neg_traces=neg_traces, aps=aps, kind=args["t"], out_path=args["o"])

if __name__ == "__main__":
    args = parser()
    args = vars(args)

    main(args)

    exit(0)



# def generate_mining_traces(args):

#     tsl = Path(args["tsl"]).read_text()
#     # ntsl = negate_tsl(tsl)

#     # tlsf = run_tsl("tlsf", tsl)
#     # ntlsf = run_tsl("tlsf", ntsl)

#     # ltl = run_syfco(tsl, flags=["-f", "ltl"])
#     # nltl = run_syfco(ntsl, flags=["-f", "ltl"])

#     # ltla = extract_assumptions(tlsf)

#     # Don't get APs from tsl hoa (it may crash on complex specs)
#     # We'll get them from the DFA generated in generate_traces() instead


#     # tlsf = run_tsl("tlsf", tsl)
#     # # hoatsl = run_tsl("hoa", Path(args["tsl"])
#     # ltl = run_syfco(tlsf, flags=["-f", "ltl"])
#     # ltln = run_neg_ltl(ltl)

#     # ltlf = run_ltlf(ltl)
#     # ltlnf = run_ltlf(ltln)

#     # ltlfhoa = run_to_finite(run_ltl2tgba(ltlf, flags=["-B", "--ltlf"]))
#     # ltlnfhoa = run_to_finite(run_ltl2tgba(ltlnf, flags=["-B", "--ltlf"]))
#     # aps = get_ap_list(ltlfhoa)

#     # assumptions = " & ".join(
#     #     [l.strip("\n") for l in extract_block(tlsf, "ASSUME").split(";") if l.strip("\n")]).replace(
#     #         "&&", "&").replace(
#     #             "||", "|")
#     # if assumptions.strip() == "":
#     #     print("No assumptions found. Supposing True.")
#     #     hoaa = run_ltl2tgba("True")
#     #     apsa = []
#     # else :

#     #     ltlfa = run_ltlf(assumptions)
#     #     hoaa = run_to_finite(run_ltl2tgba(ltlfa, flags=["-B", "--ltlf"]))
#     #     apsa = get_ap_list(hoaa)

#     # print("LTLf:", ltlf)
#     # print("!LTLf:", ltlnf)
#     # # print("HOAtsl:\n", hoatsl)
#     # print("HOAltl:\n", ltlfhoa)
#     # print("HOAltl:\n", ltlnfhoa)
#     # print("Assumptions:\n", assumptions)
#     # print("Assumption HOA:\n", hoaa)
#     # print("Atomic Propositions:", aps)
#     # print("Assumption Atomic Propositions:", apsa)


#     pos_traces, aps = generate_traces(num=args['p'],
#                                             length=args['l'],
#                                             positive=True,
#                                             tsl=tsl,
#                                             timeout=args["timeout"])
#     # print("[tracer] positive traces generated")

#     neg_traces, _ = generate_traces(num=args['n'],
#                                           length=args['l'],
#                                           positive=False,
#                                           tsl=tsl,
#                                           timeout=args["timeout"])

#     print("* pos_traces:")
#     for t in pos_traces:
#         for l in t.trace():
#             print("\t", l)
#         print(t.to_spot())
#         print("-----")
#     print("* neg_traces_vars")
#     for t in neg_traces:
#         for l in t.trace():
#             print("\t", l)
#         print(t.to_spot())
#         print("-----")

#     pos_traces_str = [t.to_spot() if args["t"] == "spot" else t.to_scarlet() for t in pos_traces]
#     neg_traces_str = [t.to_spot() if args["t"] == "spot" else t.to_scarlet() for t in neg_traces]
    
    
#     # print(f"Generated {len(pos_traces_str)} positive and {len(list(neg_traces_str))} negative traces.")

#     write_trace_file(pos_traces=pos_traces_str, 
#                      neg_traces=neg_traces_str, 
#                      kind=args["t"], 
#                      aps=aps,
#                      out_path=args["o"])


# def eval_mined_automata(args):
#     # assert args["tslm"] is not None, "Mined HOA file must be provided in eval context"

#     # tlsf = run_tsl("tlsf", Path(args["tsl"]))
#     # hoa = run_tsl("hoa", Path(args["tsl"]))
#     # aps = get_ap_list(hoa)

#     # tlsf = run_tsl("tlsf", Path(args["tsl"]))

#     # tlsf_guarantees = [l.strip("\n") for l in extract_block(tlsf, "GUARANTEE").split(";") if l.strip("\n")]

#     # tlsf_guarantees = 

#     tsl = Path(args["tsl"]).read_text()
#     # ntsl = negate_tsl(tsl)

#     # tlsf = run_tsl("tlsf", tsl)
#     # ntlsf = run_tsl("tlsf", ntsl)

#     # ltl = run_syfco(tlsf, flags=["-f", "ltl"])
#     # nltl = run_syfco(ntlsf, flags=["-f", "ltl"])

#     # ltla = extract_assumptions(tlsf)


#     # mtlsf = run_tsl("tlsf", Path(args["tslm"]))
#     # hoatsl = run_tsl("hoa", Path(args["tsl"])
#     # ltl = run_syfco(tlsf, flags=["-f", "ltl"])
#     # mltl = Path(args["tslm"]).read_text()
#     # ltln = run_neg_ltl(ltl)

#     # ltlf = run_ltlf(ltl)
#     # mltlf = run_ltlf(mltl)
#     # ltlnf = run_ltlf(ltln)

#     # ltlfhoa = run_to_finite(run_ltl2tgba(ltlf, flags=["-B", "--ltlf"]))
#     # mltlfhoa = run_to_finite(run_ltl2tgba(mltlf, flags=["-B", "--ltlf"]))
#     # ltlnfhoa = run_to_finite(run_ltl2tgba(ltlnf, flags=["-B", "--ltlf"]))

#     # aps = get_ap_list(ltlfhoa)
#     # apsm = get_ap_list(mltlfhoa)



#     # # hoam = open(args["hoam"], "r").read()
#     # # print(hoam)
#     # # hoam = run_tsl("hoa", Path(args["tslm"]))
#     # # apsm = get_ap_list(hoam)
    
#     # assumptions = " & ".join(
#     #     [l.strip("\n") for l in extract_block(tlsf, "ASSUME").split(";") if l.strip("\n")]).replace(
#     #         "&&", "&").replace(
#     #             "||", "|")
#     # if assumptions.strip() == "":
#     #     print("No assumptions found. Will not be able to validate traces.")
#     #     hoaa = run_ltl2tgba("True")
#     #     apsa = []
#     # else :
#     #     ltlfa = run_ltlf(assumptions)
#     #     hoaa = run_to_finite(run_ltl2tgba(ltlfa, flags=["-B", "--ltlf"]))
#     #     apsa = get_ap_list(hoaa)

#     # hoaa = run_ltl2tgba(assumptions)
#     # apsa = get_ap_list(hoaa)

#     pos_traces_vars, _ = generate_traces(num=args['p'], 
#                                       length=args['l'], 
#                                       positive=True,
#                                       tsl=tsl,
#                                       timeout=args["timeout"])
#     # print("[tracer] positive traces generated")

#     neg_traces_vars, _ = generate_traces(num=args['n'], 
#                                       length=args['l'], 
#                                       positive=False,
#                                       tsl=tsl,
#                                       timeout=args["timeout"])
    

#     print("pos_traces_vars:")
#     # print(pos_traces_vars)
#     for t in pos_traces_vars:
#         for l in t:
#             print("\t", l)
#         # print(trace_to_str(t, "spot"))
#         print("-----")
#     print("neg_traces_vars")
#     for t in neg_traces_vars:
#         for l in t:
#             print("\t", l)
#         # print(trace_to_str(t, "spot"))
#         print("-----")
    
    
#     mltl = Path(args["tslm"]).read_text()
#     # mltl = run_syfco(Path(args["tslm"]).read_text(), flags=["-f", "ltl"])
#     mhoa = run_ltlf2dfa(mltl)
#     apsm = get_ap_list(mhoa)
#     mdot = run_dot_gen(mhoa)
#     mmachine = Dotomata.load_dot(mdot)

#     # pos_traces_vars = generate_traces(num=args['p'], length=args['l'], aps=aps, hoa=ltlfhoa, hoaa=hoaa, apsa=apsa, positive=True, 
#     #                                   timeout=args["timeout"])
#     # neg_traces_vars = generate_traces(num=args['n'], length=args['l'], aps=aps, hoa=ltlnfhoa, hoaa=hoaa, apsa=apsa, positive=False,timeout=args["timeout"])

#     pos_traces_m = [convert_trace_to_sub_alphabet(t, apsm) for t in pos_traces_vars]
#     neg_traces_m = [convert_trace_to_sub_alphabet(t, apsm)for t in neg_traces_vars]

    

#     # TODO: dot machine check instead?
#     for i, t in enumerate(pos_traces_m):
#         if not Dotomata.check_trace_acceptance_dot(mmachine, t):
#             print("positive trace rejected:")
#             for l in pos_traces_vars[i]:
#                 print(l)
#             print(trace_to_str(pos_traces_vars[i], "spot"))
#             print("----")

#     for i, t in enumerate(neg_traces_m):
#         if Dotomata.check_trace_acceptance_dot(mmachine, t):
#             print("negative trace accepted:")
#             for l in neg_traces_vars[i]:
#                 print(l)
#             print(trace_to_str(neg_traces_vars[i], "spot"))
#             print("----")


#     mined_pos = sum(Dotomata.check_trace_acceptance_dot(mmachine, t) for t in pos_traces_m)
#     mined_neg = sum(not Dotomata.check_trace_acceptance_dot(mmachine, t) for t in neg_traces_m)


#     print(f"Evaluation results for mined automata {args['tslm']}:")
#     print(f"Positive traces accepted: {mined_pos}/{len(pos_traces_vars)}")
#     print(f"Negative traces rejected: {mined_neg}/{len(neg_traces_vars)}")
#     print(f"Total accuracy: {(mined_pos + mined_neg)/(len(pos_traces_vars) + len(neg_traces_vars)):.2f}")
