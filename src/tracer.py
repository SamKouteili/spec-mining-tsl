import argparse
import time
from pathlib import Path

from runner import *
from utils import *
from futils import *
from dotomata import Dotomata


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=str, 
                        help="Context: mine or eval")
    parser.add_argument("--tsl", type=str, required=True,
                        help="Input TSL File (for Atomic Propositions)")
    parser.add_argument("--tslm", type=str, required=False,
                        help="Input Mined TSL File (only for eval context)")
    parser.add_argument("--ltlm", type=str, required=False)
    # parser.add_argument("--hoax", type=str, required=True,
    #                     help="Input Hoax Execution Directory")
    parser.add_argument("-o", type=str,
                        help="Output trace file")
    parser.add_argument("-t", type=str, default="spot", choices=["scarlet", "spot"],
                        help="Type of trace to generate")
    parser.add_argument("-p", type=int, default=10,
                        help="Number of positive traces to generate")
    parser.add_argument("-n", type=int, default=10,
                        help="Number of negative traces to generate")
    parser.add_argument("-l", type=int, default=5,
                        help="Generated trace length")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Trace generation timeout (in seconds)")
    return parser.parse_args()


def extract_assumptions(f: str, delim="ASSUME") -> str :
    return " & ".join([l.strip("\n") for l in extract_block(f, delim).split(";") if l.strip("\n")]).replace(
        "&&", "&").replace("||", "|").strip()


def update_mutex_consistent(spot_trace: Trace) -> bool:
    """Check if a spot trace is valid (mutual exclusion of updates preserved).
        Suppose trace is in spot format."""

    # find all update terms
    update_terms = set()
    for ap in spot_trace[0]:
        if ap.strip("!").startswith("u0"):
            base_ap = ap.strip("!").strip("u0").split("0")[0]
            update_terms.add(base_ap)
    
    for letter in spot_trace:
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

# def is_valid_trace(spot_trace: Trace, assume_hoa: Path) -> bool:
#     """Check if a spot trace is valid: 
#         mutual exclusion of updates preserved and adheres to assumptions."""
#     return update_mutex(spot_trace) and 



def generate_traces(num, length, positive, tsl, timeout=10) -> tuple[list[Trace],list[str]]:
    """Generate traces for some ltl specification that adhere to some assumption ltl.
        Positive traces are accepted by both the hoa and the assumption hoa.
        return: list[traces], aps
        """
    # Use ltlf2dfa for clean DFA generation with finite trace semantics
    # ltl_ = ltl if positive else run_neg_ltl(ltl)

    tlsf = run_tsl("tlsf", tsl)
    ltl = run_syfco(tlsf, flags=["-f", "ltl"])

    # # For negative traces, also keep the positive DFA for checking
    # machinep, apsp = None, []
    # if not positive:
    #     hoap = run_ltlf2dfa(ltl) # save the positive hoa
    #     apsp = get_ap_list(hoap)
    #     dotp = run_dot_gen(hoap)
    #     machinep = Dotomata.load_dot(dotp)
    #     ntsl = negate_tsl(tsl)
    #     ntlsf = run_tsl("tlsf", ntsl)
    #     ltl = run_syfco(ntlsf, flags=["-f", "ltl"])
    
    hoa = run_ltlf2dfa(ltl)
    aps = get_ap_list(hoa)
    print("[tracer] TSL:\n",tsl)
    print("[tracer] LTL:\n",ltl)
    print("[tracer] HOA:\n",hoa)
    print("[tracer] APs:", aps)

    dot = run_dot_gen(hoa)
    machine = Dotomata.load_dot(dot)


    ltla = extract_assumptions(tlsf)
    if ltla == "":
        print("[tracer] Generating traces... no assumptions found. Supposing True.")
        hoaa = run_ltl2tgba("True")
        apsa = []
    else :
        # Use ltl2tgba for assumptions (not ltlf2dfa) because assumptions often contain
        # temporal operators like X, W, R that are not supported in LTLf
        hoaa = run_ltl2tgba(ltla, flags=["-M", "-D", "-H"])
        # print("AssUMPTION HOA:\n", hoaa)
        apsa = get_ap_list(hoaa)
    
    print("[tracer] Assumption HOA:\n", hoaa)
    print("[tracer] Assumption APs:", apsa)
    

    # dota = run_dot_gen(hoaa)
    # dota_exec = Dotomata.load_dot(dota)

    traces_vars_spot = []
    start_time = time.time()
    while len(traces_vars_spot) < num and (time.time() - start_time < timeout)  :

        trace_orig = Dotomata.ex_dot_machine(machine, length=length, cycle=True)
        trace_str = Dotomata.machine_to_spot(trace_orig, aps)
        trace_vars = str_to_trace(trace_str)

        # Debug: track final state reached
        # trace_steps = [s for s in trace_orig.split(";") if s.strip() and not s.startswith("cycle")]
        # final_state = Dotomata.walk_to_final_state(dot_exec, trace_steps)
        # if final_state in dot_exec.get("accepting", set()):
        #     accepting_state_counts[final_state] = accepting_state_counts.get(final_state, 0) + 1

        if trace_vars not in traces_vars_spot:

            trace_vars_a = convert_trace_to_sub_alphabet(trace_vars, apsa)
            if update_mutex_consistent(trace_vars) \
                and run_accept_word(hoaa, trace_to_str(trace_vars_a, "spot")): 
                # and Dotomata.check_trace_acceptance_dot(dota_exec, trace_vars_a): # TODO: dot or hoa check?
                # and run_accept_word(hoaa, trace_to_str(trace_vars_a, "spot")): 
                # print(f"[tracer] {positive} candidate: {trace_str}")
                # Check acceptance in main automaton
                # if positive or not run_accept_word(hoafp, trace_str) :
                print(f"[tracer] candidate: {trace_to_str(trace_vars, "spot")}")

                if positive :
                    if Dotomata.check_trace_acceptance_dot(machine, trace_vars, debug=False) :
                        traces_vars_spot.append(trace_vars)
                else:
                    # print(aps)
                    # print(apsp)
                    # if len(aps) != len(apsp) : # negation deleted some variable
                    #     dif_aps = list(set(apsp).difference(set(aps)))
                    #     print("dif_aps:", dif_aps)
                    #     trace_vars = add_aps_to_trace(trace_vars, dif_aps)
                    if not Dotomata.check_trace_acceptance_dot(machine, trace_vars, debug=False):
                        traces_vars_spot.append(trace_vars)
                # else:
                #     print(f"[tracer] rejected by main automaton")
                #     for l in trace_vars:
                #         print("\t", l)
                #     # print(trace_to_str(trace, "spot"))
                #     print(trace_str)
                #     print("-----")


                # else :
                #     dothoa = run_dot_gen(hoa) # original hoa (not complement hoa_)
                #     dot_machine_hoa = Dotomata.load_dot(dothoa)
                #     # Parse trace_orig to remove cycle{} notation
                #     trace_steps = [s for s in trace_orig.split(";") if s.strip() and not s.startswith("cycle")]
                #     if not Dotomata.check_trace_acceptance_dot(dot_machine_hoa, trace_steps) :
                #         print(f"[accept] {trace_to_str(trace_vars, 'spot')}")
                #         # Because complement is co-Büchi might generate trace accepted by both. Explicit final check.
                #         traces_vars_spot.append(trace_vars)
    

    assert num == 0 or traces_vars_spot, "Unable to generate any traces"

    print(f"[tracer] Generated {len(traces_vars_spot)} {positive} traces")

    return traces_vars_spot, aps

def write_trace_file(pos_traces: list[str], 
                     neg_traces: list[str], 
                     kind: str, 
                     aps: list[str],
                     out_path: str | None):
    s = ""
    for trace in pos_traces:
        s += trace + "\n"
    if len(neg_traces) > 0 or kind == "scarlet":
        s += "---\n"
    for trace in neg_traces:
        s += trace + "\n"

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

def generate_mining_traces(args):

    tsl = Path(args["tsl"]).read_text()
    # ntsl = negate_tsl(tsl)

    # tlsf = run_tsl("tlsf", tsl)
    # ntlsf = run_tsl("tlsf", ntsl)

    # ltl = run_syfco(tsl, flags=["-f", "ltl"])
    # nltl = run_syfco(ntsl, flags=["-f", "ltl"])

    # ltla = extract_assumptions(tlsf)

    # Don't get APs from tsl hoa (it may crash on complex specs)
    # We'll get them from the DFA generated in generate_traces() instead


    # tlsf = run_tsl("tlsf", tsl)
    # # hoatsl = run_tsl("hoa", Path(args["tsl"])
    # ltl = run_syfco(tlsf, flags=["-f", "ltl"])
    # ltln = run_neg_ltl(ltl)

    # ltlf = run_ltlf(ltl)
    # ltlnf = run_ltlf(ltln)

    # ltlfhoa = run_to_finite(run_ltl2tgba(ltlf, flags=["-B", "--ltlf"]))
    # ltlnfhoa = run_to_finite(run_ltl2tgba(ltlnf, flags=["-B", "--ltlf"]))
    # aps = get_ap_list(ltlfhoa)

    # assumptions = " & ".join(
    #     [l.strip("\n") for l in extract_block(tlsf, "ASSUME").split(";") if l.strip("\n")]).replace(
    #         "&&", "&").replace(
    #             "||", "|")
    # if assumptions.strip() == "":
    #     print("No assumptions found. Supposing True.")
    #     hoaa = run_ltl2tgba("True")
    #     apsa = []
    # else :

    #     ltlfa = run_ltlf(assumptions)
    #     hoaa = run_to_finite(run_ltl2tgba(ltlfa, flags=["-B", "--ltlf"]))
    #     apsa = get_ap_list(hoaa)

    # print("LTLf:", ltlf)
    # print("!LTLf:", ltlnf)
    # # print("HOAtsl:\n", hoatsl)
    # print("HOAltl:\n", ltlfhoa)
    # print("HOAltl:\n", ltlnfhoa)
    # print("Assumptions:\n", assumptions)
    # print("Assumption HOA:\n", hoaa)
    # print("Atomic Propositions:", aps)
    # print("Assumption Atomic Propositions:", apsa)


    pos_traces_vars, aps = generate_traces(num=args['p'],
                                            length=args['l'],
                                            positive=True,
                                            tsl=tsl,
                                            timeout=args["timeout"])
    # print("[tracer] positive traces generated")

    neg_traces_vars, _ = generate_traces(num=args['n'],
                                          length=args['l'],
                                          positive=False,
                                          tsl=tsl,
                                          timeout=args["timeout"])

    print("pos_traces_vars:")
    for t in pos_traces_vars:
        for l in t:
            print("\t", l)
        print(trace_to_str(t, "spot"))
        print("-----")
    print("neg_traces_vars")
    for t in neg_traces_vars:
        for l in t:
            print("\t", l)
        print(trace_to_str(t, "spot"))
        print("-----")
    

    if args["t"] == "scarlet":
        pos_traces_vars = [spot_to_scarlet_trace(trace) for trace in pos_traces_vars]
        neg_traces_vars = [spot_to_scarlet_trace(trace) for trace in neg_traces_vars]

    pos_traces_str = [trace_to_str(trace, args["t"]) for trace in pos_traces_vars]
    neg_traces_str = [trace_to_str(trace, args["t"]) for trace in neg_traces_vars]
    
    # print(f"Generated {len(pos_traces_str)} positive and {len(list(neg_traces_str))} negative traces.")

    write_trace_file(pos_traces=pos_traces_str, 
                     neg_traces=neg_traces_str, 
                     kind=args["t"], 
                     aps=aps,
                     out_path=args["o"])


def eval_mined_automata(args):
    # assert args["tslm"] is not None, "Mined HOA file must be provided in eval context"

    # tlsf = run_tsl("tlsf", Path(args["tsl"]))
    # hoa = run_tsl("hoa", Path(args["tsl"]))
    # aps = get_ap_list(hoa)

    # tlsf = run_tsl("tlsf", Path(args["tsl"]))

    # tlsf_guarantees = [l.strip("\n") for l in extract_block(tlsf, "GUARANTEE").split(";") if l.strip("\n")]

    # tlsf_guarantees = 

    tsl = Path(args["tsl"]).read_text()
    # ntsl = negate_tsl(tsl)

    # tlsf = run_tsl("tlsf", tsl)
    # ntlsf = run_tsl("tlsf", ntsl)

    # ltl = run_syfco(tlsf, flags=["-f", "ltl"])
    # nltl = run_syfco(ntlsf, flags=["-f", "ltl"])

    # ltla = extract_assumptions(tlsf)


    # mtlsf = run_tsl("tlsf", Path(args["tslm"]))
    # hoatsl = run_tsl("hoa", Path(args["tsl"])
    # ltl = run_syfco(tlsf, flags=["-f", "ltl"])
    # mltl = Path(args["tslm"]).read_text()
    # ltln = run_neg_ltl(ltl)

    # ltlf = run_ltlf(ltl)
    # mltlf = run_ltlf(mltl)
    # ltlnf = run_ltlf(ltln)

    # ltlfhoa = run_to_finite(run_ltl2tgba(ltlf, flags=["-B", "--ltlf"]))
    # mltlfhoa = run_to_finite(run_ltl2tgba(mltlf, flags=["-B", "--ltlf"]))
    # ltlnfhoa = run_to_finite(run_ltl2tgba(ltlnf, flags=["-B", "--ltlf"]))

    # aps = get_ap_list(ltlfhoa)
    # apsm = get_ap_list(mltlfhoa)



    # # hoam = open(args["hoam"], "r").read()
    # # print(hoam)
    # # hoam = run_tsl("hoa", Path(args["tslm"]))
    # # apsm = get_ap_list(hoam)
    
    # assumptions = " & ".join(
    #     [l.strip("\n") for l in extract_block(tlsf, "ASSUME").split(";") if l.strip("\n")]).replace(
    #         "&&", "&").replace(
    #             "||", "|")
    # if assumptions.strip() == "":
    #     print("No assumptions found. Will not be able to validate traces.")
    #     hoaa = run_ltl2tgba("True")
    #     apsa = []
    # else :
    #     ltlfa = run_ltlf(assumptions)
    #     hoaa = run_to_finite(run_ltl2tgba(ltlfa, flags=["-B", "--ltlf"]))
    #     apsa = get_ap_list(hoaa)

    # hoaa = run_ltl2tgba(assumptions)
    # apsa = get_ap_list(hoaa)

    pos_traces_vars, _ = generate_traces(num=args['p'], 
                                      length=args['l'], 
                                      positive=True,
                                      tsl=tsl,
                                      timeout=args["timeout"])
    # print("[tracer] positive traces generated")

    neg_traces_vars, _ = generate_traces(num=args['n'], 
                                      length=args['l'], 
                                      positive=False,
                                      tsl=tsl,
                                      timeout=args["timeout"])
    

    print("pos_traces_vars:")
    # print(pos_traces_vars)
    for t in pos_traces_vars:
        for l in t:
            print("\t", l)
        # print(trace_to_str(t, "spot"))
        print("-----")
    print("neg_traces_vars")
    for t in neg_traces_vars:
        for l in t:
            print("\t", l)
        # print(trace_to_str(t, "spot"))
        print("-----")
    
    
    mltl = Path(args["tslm"]).read_text()
    # mltl = run_syfco(Path(args["tslm"]).read_text(), flags=["-f", "ltl"])
    mhoa = run_ltlf2dfa(mltl)
    apsm = get_ap_list(mhoa)
    mdot = run_dot_gen(mhoa)
    mmachine = Dotomata.load_dot(mdot)

    # pos_traces_vars = generate_traces(num=args['p'], length=args['l'], aps=aps, hoa=ltlfhoa, hoaa=hoaa, apsa=apsa, positive=True, 
    #                                   timeout=args["timeout"])
    # neg_traces_vars = generate_traces(num=args['n'], length=args['l'], aps=aps, hoa=ltlnfhoa, hoaa=hoaa, apsa=apsa, positive=False,timeout=args["timeout"])

    pos_traces_m = [convert_trace_to_sub_alphabet(t, apsm) for t in pos_traces_vars]
    neg_traces_m = [convert_trace_to_sub_alphabet(t, apsm)for t in neg_traces_vars]

    

    # TODO: dot machine check instead?
    for i, t in enumerate(pos_traces_m):
        if not Dotomata.check_trace_acceptance_dot(mmachine, t):
            print("positive trace rejected:")
            for l in pos_traces_vars[i]:
                print(l)
            print(trace_to_str(pos_traces_vars[i], "spot"))
            print("----")

    for i, t in enumerate(neg_traces_m):
        if Dotomata.check_trace_acceptance_dot(mmachine, t):
            print("negative trace accepted:")
            for l in neg_traces_vars[i]:
                print(l)
            print(trace_to_str(neg_traces_vars[i], "spot"))
            print("----")


    mined_pos = sum(Dotomata.check_trace_acceptance_dot(mmachine, t) for t in pos_traces_m)
    mined_neg = sum(not Dotomata.check_trace_acceptance_dot(mmachine, t) for t in neg_traces_m)


    print(f"Evaluation results for mined automata {args['tslm']}:")
    print(f"Positive traces accepted: {mined_pos}/{len(pos_traces_vars)}")
    print(f"Negative traces rejected: {mined_neg}/{len(neg_traces_vars)}")
    print(f"Total accuracy: {(mined_pos + mined_neg)/(len(pos_traces_vars) + len(neg_traces_vars)):.2f}")




if __name__ == "__main__":
    args = parser()
    args = vars(args)

    if args["context"] == "mine":
        generate_mining_traces(args)
    elif args["context"] == "eval":
        eval_mined_automata(args)
    else :
        print(f"Unknown context {args['context']}. Use 'mine' or 'eval'.")
        exit(1)


    exit(0)
