# run_scarlett.py
from Scarlet.ltllearner import LTLlearner
import multiprocessing as mp
import os
import csv
import argparse
from pathlib import Path
import time

from runners import run_tsl, run_dot_gen, run_ltlf2dfa
from utils import *
from tracer import Trace
from dotomata import Dotomata

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="Input Scarlet Trace")
    parser.add_argument('-o', type=str, required=True, help="Output TSL path")
    parser.add_argument("--tsl", type=str, help="Original TSL files (for assumptions)")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--csv", type=str, default="scarlet_out.csv")
    parser.add_argument("--thresh", type=int, default=0)
    return parser.parse_args()


def extract_last_formula(input_file):
    """Extract the last completed formula from scarlet_out.csv"""

    if not os.path.isfile(input_file):
        print(f"Error: Input file {input_file} not found")
        exit(1)

    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip header, find last row with 4 columns and last column is '1' (completed)
    for row in reversed(rows[1:]):  # Skip header
        if len(row) >= 4 and row[3].strip() == '1':
            formula = row[2].strip()
            return formula

    print('Error: No completed formula found')
    exit(1)

def ensure_split(trace_f, tsl_f, last_formula):
    # Sanity check
    pos_scarlet, neg_scarlet, ops, aps = [s.strip("\n") for s in Path(trace_f).read_text().split("---")]
    aps = aps.split(",")
    pos_traces, neg_traces = [[Trace.from_scarlet(t, aps) for t in scarlet.split("\n")] 
                              for scarlet in [pos_scarlet, neg_scarlet]]

    mhoa = run_ltlf2dfa(last_formula)
    maps = get_ap_list(mhoa)
    mdot = run_dot_gen(mhoa)
    mmachine = Dotomata(mdot)


    
    # print(mhoa)

    # failed = False
    for trace in pos_traces:
        mtrace = trace.change_alphabet(maps)
        if not mmachine.accepts(mtrace):
            print('[scarlet] word from pos traces rejected:\n\t', trace, "\n\t", mtrace)
            exit(1)
            # failed = True
        # if not run_accept_word(hoam_tsl, word_tsl):
        #     print('[tsl hoa] word from pos traces rejected:\n', word)

    for trace in neg_traces:
        mtrace = trace.change_alphabet(maps)
        # word = trace_to_str(convert_trace_to_sub_alphabet(trace, apsm), "spot")
        # word_tsl = trace_to_str(convert_trace_to_sub_alphabet(trace, apsm_tsl), "spot")
        if mmachine.accepts(trace.change_alphabet(maps)):
            print('[scarlet] word from pos traces rejected:\n\t', trace, "\n\t", mtrace)
            exit(1)
            # failed = True
        # if run_accept_word(hoam_tsl, word_tsl):
        #     print('[tsl hoa] word from neg traces not accepted:\n', word)
    
    # if not failed:
    #     print("yay passed")


def write_formula_to_file(formula, output_file):
    """Write formula wrapped in GUARANTEE block to output file"""

    content = f"""GUARANTEE {{
    {formula};
}}"""
    
    base_name = os.path.splitext(os.path.basename(output_file))[0]

    # with open(f'{base_name}.tlsf', 'w') as f:
    #     f.write(content)

    print("TLSF:\n", content)

    tsl = run_tsl("fromtlsf", content)

    # tsl = tsl.replace("always", "initially")

    print("TSL:\n", tsl)
 
    # os.remove(f'{base_name}.tlsf')

    with open(output_file, 'a') as f:
        f.write(tsl)

def main():
    args = parser()
    args = vars(args)

    # if os.exists()
    # os.remove(args['o'], exists) # delete old version


    start_time = time.time()
    learner = LTLlearner(
        input_file=args["i"],       # trace file
        timeout=args["timeout"],    # optional
        csvname=args["csv"],    # optional
        thres=args["thresh"]         # optional, 0 = perfect separation
    )
    learner.learn()
    print("[scarlet] total time: ", time.time() - start_time)

    if args["tsl"]:
        tsl_text = Path(args["tsl"]).read_text()
        assumptions = extract_block(tsl_text, "assume")
        if not assumptions:
            print("Unable to extract TSL assumptions.")


        else:
            content = f"""always assume {{
                {assumptions}
}}"""
            with open(args["o"], 'w') as f:
                f.write(content)

    last_formula = extract_last_formula(args["csv"])

    with open(f"{args['o']}.ltl", "w") as f:
        f.write("".join(last_formula.split("\n")))

    # Write to tsl output file
    write_formula_to_file(last_formula, args["o"])


    # print(last_formula)

    # ensure_split(args["i"], args["o"], last_formula)

    # print(pos_traces)
    # print('\n\n\n')
    # print(neg_traces)
    # print(aps)


if __name__ == "__main__":
    # On macOS + Python 3.8+, spawn is default/safest. Make it explicit.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, fine.
        pass
    main()
