"""Utils for trace processing/translation"""
import re

# Types; TODO: Class?
AP = str
Letter = list[AP]
Trace = list[Letter]

def get_ap_list(hoa: str) -> list[str]:
    """Extract the atomic propositions declared in a HOA."""
    for line in hoa.splitlines():
        if line.strip().startswith("AP:"):
            tokens = line.split()
            assert len(
                tokens) >= 3, "AP line should be formatted 'AP: <count> <proposition1> <proposition2> ...'"
            # tokens[0] == "AP:", tokens[1] == count, rest are proposition names
            return [token.strip('"') for token in tokens[2:]]
    return []


def convert_hoax_to_trace(hoax_out: str, aps, kind) -> Trace:
    """Generate trace list from hoax output (Spot/Scarlet)."""
    trace = []
    for line in hoax_out.splitlines():
        raw = re.search(r"{(.*)}", line)
        if not raw:
            continue
        present = [tok.strip("'\" ")
                   for tok in raw.group(1).split(",") if tok.strip()]
        
        assignment = ["1" if ap in present else "0" for ap in aps] \
              if kind == "scarlet" else \
              [ap if ap in present else f"!{ap}" for ap in aps]
        trace.append(assignment)

    return trace


# def generate_random_spot_trace(aps: list[str], length: int) -> list[list[str]]:
#     """Generate a random trace of given length and type (spot).
#         Random trace does not violate mutex of updates."""
#     trace = []
#     while len(trace) < length:
#         letter = [ap if random.choice([True, False]) else f"!{ap}" for ap in aps]
#             # if random.choice([True, False]):
#             #     letter.append(ap)
#             # else:
#             #     letter.append(f"!{ap}" if kind == "spot" else "0")
#         # assumption_trace = get_trace_string(convert_trace_to_sub_alphabet(trace + [letter], apsa), "spot")
#         if update_mutex([letter]) :
#             trace.append(letter)
#     return trace

def trace_to_str(trace_list: Trace, kind: str) -> str:
    """Convert a trace list to a string representation."""
    if not trace_list or not all(trace_list):
        # print("[trace_to_str] empty trace presented")
        return "cycle{1}" if kind == "spot" else ""
    trace_str = []
    for trace in trace_list:
        if kind == "scarlet":
            trace_str.append(",".join(trace))
        elif kind == "spot":
            trace_str.append(" & ".join(trace))
    return ";".join(trace_str) + (";cycle{1}" if kind == "spot" else "")

def str_to_trace(trace_str: str, kind: str = "spot") -> Trace:
    """Convert a trace string to a Trace (list[list[str]])."""
    split_var = "&" if kind == "spot" else ","
    return [[ap.strip() for ap in step.strip().split(split_var) if ap.strip()] 
            for step in trace_str.split(";") if step.strip() and not step.startswith("cycle{")]

def convert_trace_to_sub_alphabet(trace : Trace, aps : list[str]) -> list[list[str]]:
    """Convert a trace to a subset alphabet of atomic propositions. 
        Assume same type of trace (spot). Assume new alphabet is a subset of original alphabet."""
    return [[ap for ap in letter if ap.strip("!") in aps] for letter in trace]

def spot_to_scarlet_trace(spot_trace: Trace) -> Trace:
    """Convert a spot trace to scarlet trace."""
    return [["0" if ap.startswith("!") else "1" for ap in letter] for letter in spot_trace]

def scarlet_to_spot_trace(scarlet_trace: Trace, aps: list[str]) -> Trace:
    """Convert a scarlet trace to a spot trace"""
    return [[aps[i] if int(ap) == 1 else f"!{aps[i]}" for i, ap in enumerate(letter)] for letter in scarlet_trace]


# def flip_variable(trace: Trace, i, j):
#     """Flip the variable at the given index in the trace."""
#     if trace[i][j] == "1":
#         trace[i][j] = "0"
#     elif trace[i][j] == "0":
#         trace[i][j] = "1"
#     elif trace[i][j].startswith("!"):
#         trace[i][j] = trace[i][j][1:]  # Remove '!' to make it positive
#     else:
#         trace[i][j] = "!" + trace[i][j]  # Add '!' to make it negative
#     return trace

