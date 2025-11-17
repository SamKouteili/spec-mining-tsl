# FIRST: Create original table, applying functions on updates and predicates, also adding END
# SECOND: Check for any point (not END) where NO update APs are true on a variable, we will then need to consider function compositions
# THIRD: For each variable, check if updated more than once at given timestep, if so, need to create seperate datasets with different instances and take union

from argparse import ArgumentParser
from pathlib import Path
from tracer import Trace
import pandas as pd
from itertools import product
import json
from copy import deepcopy

def parse_args():
    parser = ArgumentParser(description="Process log data to boolean trace format")
    parser.add_argument("traces", type=Path, help="Directory of pos/neg traces")
    parser.add_argument("meta", type=Path,  help="Trace metadata (functions, predicates, types)")
    return parser.parse_args()

# Redundant class could have done it with just Function but this is more explicit
# class Predicate:
#     def __init__(self, name: str, arg_types: str, impl):
#         self.name = name
#         self.input_types = arg_types.split("->")[:-1]
#         self.p = impl

#     def __str__(self):
#         return f"{self.name}: {" * ".join(self.input_types)}"
    
#     def run(self, *args):
#         return self.p(*args)

class Function:
    def __init__(self, name: str, arg_types: str, impl):
        self.name = name
        self.input_types = arg_types.split("->")[:-1]
        self.output_type = arg_types.split("->")[-1]
        self.f = impl

    def __str__(self):
        return f"{self.name}: {" * ".join(self.input_types)} -> {self.output_type})"
    
    def run(self, *args):
        return self.f(*args)
    

def import_metadata(meta_path: Path) -> dict[str, dict]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("meta", str(meta_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metadata module from {meta_path}")
    meta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meta)

    v, f, p = None, None, None
    try:
        v = meta.VARS
        f = {name: Function(name, *details) for name, details in meta.FUNCTIONS.items()}
        p = {name: Function(name, *details) for name, details in meta.PREDICATES.items()}
    except AttributeError as e:
        raise AttributeError(f"Missing expected attribute in metadata: {e}")
    
    # assert v and f and p, "Metadata must contain VARS, FUNCTIONS, and PREDICATES"

    return {
        "vars": v,
        "functions": f,
        "predicates": p
    }

    
class Update:
    def __init__(self, var: str, term: str):
        self.var = var
        self.term = term


    def __str__(self):
        return f"[{self.var} <- {self.term}]"
    
    def __eq__(self, other):
        return isinstance(other, Update) and self.var == other.var and self.term == other.term
    
    def __hash__(self):
        return hash((self.var, self.term))

class UpdateF:
    def __init__(self, var: str, func, inputs: tuple[str]):
        self.var = var
        self.func = func
        self.inputs = inputs

    def __str__(self):
        return f"[{self.var} <- {self.func.name} {' '.join(self.inputs)}]"
    
    def __eq__(self, other):
        return isinstance(other, UpdateF) and self.var == other.var and self.func.name == other.func.name and self.inputs == other.inputs
    
    def __hash__(self):
        return hash((self.var, self.func.name, self.inputs))

class Predicate:
    def __init__(self, pred: Function, inputs: tuple[str]):
        self.pred = pred
        self.inputs = inputs

    def __str__(self):
        return f"{self.pred.name} {' '.join(self.inputs)}"
    
    def __eq__(self, value: object) -> bool:
        return isinstance(value, Predicate) and self.pred.name == value.pred.name and self.inputs == value.inputs
    
    def __hash__(self):
        return hash((self.pred.name, self.inputs))
    

Table = list[dict[Update | UpdateF | Predicate, bool]]

def build_ap_table(log, metadata) -> Table:
    """Build AP table for a given trace and metadata.
    
        e.g. Suppose log:
        {"ball": 1}
        {"ball": 2}
        {"ball": 1}
        {"ball": 0}
        {"ball": -1}
        {"ball": -2}

        And metadata:
        VARS = {
            "ball": int
        }
        FUNCTIONS = {
            "moveRight": ("int->int", lambda ball: ball + 1),
            "moveLeft": ("int->int", lambda ball: ball - 1),
        }
        PREDICATES = {
            "rightMost": ("int->bool", lambda ball: ball == -2),
            "leftMost": ("int->bool", lambda ball: ball == 2),
        }

        Then AP table would be:
        [
            {"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": False, "END": False},
            {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": True,  "leftMost ball": False, "END": False},
            {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False},
            {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False},
            {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False},
            {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": True,  "END": True}
        ]
    """
    ap_table = []

    for i, entry in enumerate(log):

        table_entry = {}
        for var_name, _ in metadata["vars"].items():
            # Identity APs and simple updates for variable
            for var2_name, _ in metadata["vars"].items():
                if metadata["vars"][var_name] == metadata["vars"][var2_name]: # same type
                    ap = Update(var_name, var2_name)
                    next_val = log[i+1][var_name] if i + 1 < len(log) else None
                    table_entry[ap] = entry[var2_name] == next_val if next_val is not None else False
    
        for _, func in metadata["functions"].items():
            # Permutations (with repetition) of function arguments applied to all variables
            # print(f"[biuld_ap_table] combs={list(product(metadata['vars'].keys(), repeat=len(func.input_types)))}")
            for var_comb in product(metadata["vars"].keys(), repeat=len(func.input_types)):
                # Check if types match
                if [metadata["vars"][var] for var in var_comb] == func.input_types:
                    # print(f"[ap_table] func={func}, var_comb={var_comb}")
                    # print(f"[ap_table] var_comb={var_comb}")

                    # Create AP for each variable which the function is applied to.
                    # Check output type matches variable type
                    for var in metadata["vars"].keys():
                        if metadata["vars"][var] == func.output_type:
                            ap = UpdateF(var, func, var_comb)
                            next_val = log[i+1][var] if i + 1 < len(log) else None
                            table_entry[ap] = func.run(*[entry[v] for v in var_comb]) == next_val if next_val is not None else False
            
        for _, pred in metadata["predicates"].items():
            # Permutations (with repetition) of predicate arguments applied to all variables
            for var_comb in product(metadata["vars"].keys(), repeat=len(pred.input_types)):
                # Check if types match
                if [metadata["vars"][var] for var in var_comb] == pred.input_types:
                    ap = Predicate(pred, var_comb)
                    table_entry[ap] = pred.run(*[entry[v] for v in var_comb])
        
        table_entry["END"] = (i == len(log) - 1)

        ap_table.append(table_entry)
 
    
    return ap_table


def check_updates(ap_table: Table, vrs: dict[str, str]) -> list[dict[str, int]]:
    """Check for any point how many updates are applied to a term."""
    num_updates = []
    for entry in ap_table:
        updates_at_time_step = {var: sum(entry[ap] for ap in entry if (isinstance(ap, Update) or isinstance(ap, UpdateF)) and ap.var == var) for var in vrs.keys()}
        num_updates.append(updates_at_time_step)
    return num_updates

def copy_table(table: Table) -> Table:
    return [{ deepcopy(ap): asgn for ap, asgn in d.items() } for d in table]


# class Tree :
#     def __init__(self):



def split_tables(ap_table: Table, vrs: dict[str, str], i=0) -> list[Table]:
    """Split AP table into multiple tables if multiple updates are applied to a term at the same time step.
        e.g. Suppose at time step 0, we have:
        {"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": True, "rightMost ball": False, "leftMost ball": False, "END": False}
        Then we would need to create two tables:
        [
            {"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": False, "END": False},
            {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False}
    ]
    """

    print(f"split_tables: i={i}")

    if i == len(ap_table) - 1:
        return [ap_table]
    
    updates_at_time_step = {var: [ap for ap in ap_table[i] 
                                    if (isinstance(ap, Update) or isinstance(ap, UpdateF)) 
                                        and ap.var == var and ap_table[i][ap]] 
                                    for var in vrs.keys()}
    
    out = []
    splitted = False
    for var, asgns in updates_at_time_step.items():
        print(var, [str(k) for k in updates_at_time_step[var]])
        if len(asgns) > 1:
            splitted = True
            for asgn in asgns:
                print(f"asgn={asgn}")
                new_table = copy_table(ap_table)
                for asgn2 in asgns:
                    if asgn != asgn2:
                        new_table[i][asgn] = False
                print_table(new_table)
                # print(f"new table: {new_table}")
                out.extend(split_tables(new_table, vrs, i+1))
    
    if not splitted :
        out.extend(split_tables(ap_table, vrs, i+1))
            

    # new_tables = [copy_table(ap_table)]
    # final_tables = []

    # while new_tables != []:
    #     cur_table = new_tables.pop()
    #     for word in cur_table:
    #         updates_at_time_step = {var: [ap for ap in word 
    #                                       if (isinstance(ap, Update) or isinstance(ap, UpdateF)) and ap.var == var and word[ap]] 
    #                                       for var in vrs.keys()}
    #         for var, asgns in updates_at_time_step.items():
    #             if len(asgns) > 1:
    #                 print(var, [str(k) for k in updates_at_time_step[var]])
    #                 for asgn in asgns:
    #                     new_table = copy_table(cur_table)
    #                     for asgn2 in asgns:
    #                         if asgn != asgn2:
    #                             new_table[i][asgn] = False
    #                     # print(f"new table: {new_table}")
    #                     new_tables.append(new_table)
    
    return out

        



def print_table(ap_table: Table):
    print("\n".join(str({str(ap): val for ap, val in row.items()}) for row in ap_table))


def compose_metadata_functions(metadata):
    """Compose functions in metadata to create new functions."""
    new_functions = {}
    for f1 in metadata["functions"].values():
        for f2 in metadata["functions"].values():
            for i, ty in enumerate(f2.input_types):

                # NOTE: Only composing currently on first arg. Should be extended to multiple args
                if f1.output_type == ty:  # Check if output of f1 matches input of f2
                    composed_name = f"{f2.name}({i}{f1.name})"
                    composed_arg_types = "->".join(f2.input_types[:i] + f1.input_types + f2.input_types[i+1:] + [f2.output_type])
                    composed_impl = lambda *args, f1=f1, f2=f2: f2.run(f1.run(*args[:len(f1.input_types)]), *args[len(f1.input_types):])
                    new_functions[composed_name] = (composed_arg_types, composed_impl)
    
    metadata["functions"].update({name: Function(name, *details) for name, details in new_functions.items()})
    return metadata

# TODO: Different mining instances for multiple updates at same time step


if __name__ == "__main__":
    args = parse_args()
    print(f"Processing traces from: {args.traces}")
    print(f"Using metadata from: {args.meta}")

    metadata = import_metadata(args.meta)
    print("Metadata loaded successfully:")
    print("  Variables:", metadata['vars'])
    print("  Functions:", {str(fun) for fun in metadata['functions'].values()})
    print("  Predicates:", {str(pred) for pred in metadata['predicates'].values()})

    pos = args.traces / "pos"
    neg = args.traces / "neg"

    for trace_file in pos.glob("*.jsonl"):
        print("----------------")
        with trace_file.open("r", encoding="utf-8") as fh:
            log = [json.loads(line) for line in fh if line.strip()]
            good_tables = []
            good_trace = False 
            while not good_trace:
                table = build_ap_table(log, metadata)
                print(f"AP table for {trace_file.name}:")
                print_table(table)

                num_updates = check_updates(table, metadata['vars'])
                print(f"Number of updates at each time step for {trace_file.name}:")
                for i, updates in enumerate(num_updates):
                    print(f" {i}: {updates}")

                if any(any(update == 0 for update in updates.values()) for updates in num_updates[:-1]):
                    print("Detected table entry with no updates. Composing functions...")
                    metadata = compose_metadata_functions(metadata)
                    print("new metadata:")
                    print("  Variables:", metadata['vars'])
                    print("  Functions:", {str(fun) for fun in metadata['functions'].values()})
                    print("  Predicates:", {str(pred) for pred in metadata['predicates'].values()})
                elif any(any(update > 1 for update in updates.values()) for updates in num_updates[:-1]):
                    print("Detected table entry with multiple updates. Splitting mining traces...")
                    tables = split_tables(table, metadata['vars'])
                    print(f"Split into {len(tables)} tables:")
                    for t in tables:
                        print("----")
                        print_table(t)
                        for i, updates in enumerate(check_updates(t, metadata['vars'])):
                            print(f" {i}: {updates}")
                    good_tables.extend(tables)
                    good_trace = True
                else :
                    print("All entries have updates. No need to compose functions.")
                    good_tables.append(table)
                    good_trace = True
                    
                
                

        # 'log' is now a list of JSON objects (dicts) from the .jsonl file



    # for trace_file in args.traces.glob("*.jsonl"):
    #     print(f"Processing trace: {trace_file}")
    #     trace = Trace.from_jsonl(trace_file, metadata)
    #     out_file = trace_file.with_suffix(".trace")
    #     trace.to_trace(out_file)

