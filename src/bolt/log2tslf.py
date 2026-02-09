import json
from argparse import ArgumentParser
from pathlib import Path
from itertools import product
from copy import deepcopy


def is_tuple_value(value):
    """Check if a value is a tuple (represented as a list in JSON)."""
    return isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (int, float)) for x in value)


def is_boolean_value(value):
    """Check if a value is a boolean (true/false in JSON becomes Python bool)."""
    return isinstance(value, bool)


def expand_tuple_keys(obj: dict) -> dict:
    """
    Expand tuple values into separate element keys.
    e.g., {"player": [0, 1]} -> {"player[0]": 0, "player[1]": 1}
    Non-tuple values are kept as-is.
    """
    expanded = {}
    for key, value in obj.items():
        if is_tuple_value(value):
            for i, elem in enumerate(value):
                expanded[f"{key}[{i}]"] = elem
        else:
            expanded[key] = value
    return expanded


def reconstruct_tuple(obj: dict, tuple_name: str, arity: int) -> tuple:
    """
    Reconstruct a tuple from expanded element keys.
    e.g., {"player[0]": 0, "player[1]": 1} with tuple_name="player", arity=2 -> (0, 1)
    """
    return tuple(obj[f"{tuple_name}[{i}]"] for i in range(arity))

def parse_args():
    parser = ArgumentParser(description="Process log data to boolean trace format")
    parser.add_argument("traces", type=Path, help="Directory of pos/neg traces")
    parser.add_argument("meta", type=Path,  help="Trace metadata (functions, predicates, types)")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory for processed traces")
    parser.add_argument("--self-inputs-only", action="store_true",
                        help="Only consider self-updates (playerX <- f(playerX)), skip cross-updates")
    parser.add_argument("--pos", type=Path, help="Explicit path to positive traces directory (default: traces/pos)")
    parser.add_argument("--neg", type=Path, help="Explicit path to negative traces directory (default: traces/neg)")
    return parser.parse_args()

#################################################################
################## METADATA CLASSES #############################
#################################################################

class Variable:
    def __init__(self, name: str, var_type: str):
        self.name = name
        self.type = var_type

    def __str__(self):
        return f"{self.name}" # [{self.type}]"
    
    def __eq__(self, other):
        return isinstance(other, Variable) and (self.name == other.name and self.type == other.type)

    def __hash__(self):
        return hash((self.name, self.type))

class Function:
    def __init__(self, name: str, arg_types: str, impl, issy_template: str = None):
        self.name = name
        self.input_types = arg_types.split("->")[:-1]
        self.output_type = arg_types.split("->")[-1]
        self.f = impl
        # Issy template with placeholders like {0}, {1} for args
        # e.g., "add {0} i1()" for increment by 1
        self.issy_template = issy_template

    def __str__(self):
        return f"{self.name}[{" * ".join(self.input_types)} -> {self.output_type}]"

    # not checking implementation explicitly. could lookat bytecode if needed
    def __eq__(self, other):
        return isinstance(other, Function) and \
            self.name == other.name and \
            self.input_types == other.input_types and \
            self.output_type == other.output_type

    def __hash__(self):
        return hash((self.name, tuple(self.input_types), self.output_type))

    def run(self, *args):
        return self.f(*args)

    def format_issy(self, inputs: tuple) -> str:
        """
        Format this function application in Issy format.

        Args:
            inputs: Tuple of Variable objects to substitute for placeholders

        Returns:
            Issy-formatted string, e.g., "add x i1()"
        """
        if self.issy_template:
            # Substitute placeholders {0}, {1}, etc. with input variable names
            result = self.issy_template
            for i, inp in enumerate(inputs):
                result = result.replace(f"{{{i}}}", str(inp.name))
            return result
        else:
            # Fallback: use old style with function name
            return f"{self.name} {' '.join([str(inp.name) for inp in inputs])}"
    
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
   

class Metadata:
    def __init__(self, variables: set[Variable], functions: set[Function], predicates: set[Function],
                 tuple_vars: dict = None, boolean_vars: set[str] = None):
        self.vars = variables
        self.functions = functions
        self.predicates = predicates
        self.tuple_vars = tuple_vars or {}
        self.boolean_vars = boolean_vars or set()  # Names of boolean stream variables

    def __str__(self):
        out = "metadata {\n"
        out += "  variables {\n"
        for var in self.vars:
            out += f"    {var}\n"
        out += "  }\n"
        out += "  functions {\n"
        for func in self.functions:
            out += f"    {func}\n"
        out += "  }\n"
        out += "  predicates {\n"
        for pred in self.predicates:
            out += f"    {pred}\n"
        out += "  }\n"
        if self.tuple_vars:
            out += "  tuple_vars {\n"
            for name, info in self.tuple_vars.items():
                out += f"    {name}: arity={info['arity']}, const={info['const']}\n"
            out += "  }\n"
        if self.boolean_vars:
            out += "  boolean_vars {\n"
            for name in self.boolean_vars:
                out += f"    {name}\n"
            out += "  }\n"
        out += "}"
        return out

    @classmethod
    def import_metadata(cls, meta_path: Path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("meta", str(meta_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load metadata module from {meta_path}")
        meta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(meta)

        v, f, p = None, None, None
        tuple_vars = {}
        boolean_vars = set()
        try:
            v = set(Variable(name, ty) for name, ty in meta.VARS.items())
            f = set(Function(name, *details) for name, details in meta.FUNCTIONS.items())
            p = set(Function(name, *details) for name, details in meta.PREDICATES.items())
            # Load tuple variable metadata if available
            if hasattr(meta, 'TUPLE_VARS'):
                tuple_vars = meta.TUPLE_VARS
            # Load boolean variable names if available
            if hasattr(meta, 'BOOLEAN_VARS'):
                boolean_vars = set(meta.BOOLEAN_VARS)
        except AttributeError as e:
            raise AttributeError(f"Missing expected attribute in metadata: {e}")

        # If we have tuple_vars, add tuple-type variables to the variable set
        if tuple_vars:
            for tup_name, info in tuple_vars.items():
                arity = info['arity']
                is_const = info['const']
                tup_type = f"{'const ' if is_const else ''}tuple{arity}"
                v.add(Variable(tup_name, tup_type))

        return Metadata(variables=v, functions=f, predicates=p, tuple_vars=tuple_vars, boolean_vars=boolean_vars)
    
    # def compose_functions(self):
    #     """Compose functions in metadata to create new functions."""
    #     new_functions = {}
    #     for f1 in self.functions:
    #         for f2 in self.functions:
    #             for i, ty in enumerate(f2.input_types):
    #                 if f1.output_type == ty:  # Check if output of f1 matches input of f2
    #                     composed_name = f"{f2.name}({i}{f1.name})"
    #                     composed_arg_types = "->".join(f2.input_types[:i] + f1.input_types + f2.input_types[i+1:] + [f2.output_type])
    #                     composed_impl = lambda *args, f1=f1, f2=f2: f2.run(f1.run(*args[:len(f1.input_types)]), *args[len(f1.input_types):])
    #                     new_functions[composed_name] = Function(composed_name, composed_arg_types, composed_impl)
        
    #     self.functions.update(new_functions)

    #     return self


#################################################################
################## TSLf ATOMIC CLASSES ##########################
#################################################################

class Update:
    def __init__(self, var: Variable, func: Function, inputs: tuple[Variable, ...]):
        self.var = var
        self.func = func
        self.inputs = inputs

    @classmethod
    def from_string(cls, s: str) -> 'Update':
        """Parse an update string like '[x <- inc x]' or '[x <- x]' (identity)."""
        s = s.strip()
        if not (s.startswith('[') and s.endswith(']')):
            raise ValueError(f"Update must be wrapped in brackets: {s}")
        inner = s[1:-1].strip()

        if '<-' not in inner:
            raise ValueError(f"Update must contain '<-': {s}")

        lhs, rhs = inner.split('<-', 1)
        var_name = lhs.strip()
        rhs = rhs.strip()

        # Create the output variable (type unknown, use 'int' as default)
        var = Variable(var_name, 'int')

        # Parse RHS: could be identity (just a var) or function application
        rhs_parts = rhs.split()
        if len(rhs_parts) == 1:
            # Identity: [x <- x] or [x <- y]
            input_var = Variable(rhs_parts[0], 'int')
            func = Function("", "int->int", lambda x: x)
            return cls(var, func, (input_var,))
        else:
            # Function application: [x <- func arg1 arg2 ...]
            func_name = rhs_parts[0]
            input_vars = tuple(Variable(v, 'int') for v in rhs_parts[1:])
            # Construct type signature based on arity
            arg_types = '->'.join(['int'] * len(input_vars)) + '->int'
            func = Function(func_name, arg_types, lambda *args: None)  # Placeholder impl
            return cls(var, func, input_vars)

    def __str__(self):
        """
        Format the update in Issy-compatible TSL format.

        For functions with Issy templates:
            [x <- add x i1()]  (increment x by 1)
            [y <- sub y i1()]  (decrement y by 1)

        For identity (empty func name):
            [x <- x]

        For functions without Issy templates (fallback):
            [x <- funcname input1 input2]
        """
        # Identity case: empty function name means [var <- var]
        if not self.func.name:
            return f"[{self.var} <- {self.inputs[0]}]"

        # Use Issy template if available
        if self.func.issy_template:
            issy_expr = self.func.format_issy(self.inputs)
            return f"[{self.var} <- {issy_expr}]"

        # Fallback: old style with function name (should rarely happen with new metadata)
        return f"[{self.var} <- {self.func.name} {' '.join([str(inp) for inp in self.inputs])}]"

    def __repr__(self):
        return str(self)
        # return f"u0{self.var}0{self.term}0"

    def __eq__(self, other):
        return isinstance(other, Update) and \
            self.var == other.var and \
            self.func.name == other.func.name and \
            self.inputs == other.inputs

    def __hash__(self):
        return hash((self.var, self.func.name, self.inputs))

class Predicate:
    def __init__(self, pred: Function, inputs: tuple[Variable, ...]):
        self.pred = pred
        self.inputs = inputs

    @classmethod
    def from_string(cls, s: str) -> 'Predicate':
        """Parse a predicate string like 'eqC taxi locY' or 'eq x y'."""
        parts = s.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Predicate must have at least a name and one argument: {s}")

        pred_name = parts[0]
        input_vars = tuple(Variable(v, 'int') for v in parts[1:])

        # Construct type signature based on arity
        arg_types = '->'.join(['int'] * len(input_vars)) + '->bool'
        pred = Function(pred_name, arg_types, lambda *args: None)  # Placeholder impl
        return cls(pred, input_vars)

    def __str__(self):
        return f"{self.pred.name} {' '.join([str(inp) for inp in self.inputs])}"

    def __repr__(self):
        return str(self)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Predicate) and self.pred.name == value.pred.name and self.inputs == value.inputs

    def __hash__(self):
        return hash((self.pred.name, self.inputs))


class BooleanAP:
    """Atomic proposition for boolean stream variables (e.g., passengerInTaxi)."""
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_string(cls, s: str) -> 'BooleanAP':
        """Parse a boolean AP string like 'END' or 'passengerInTaxi'."""
        return cls(s.strip())

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, BooleanAP) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
    

# Atomic Proposition TSLf Table
class APTable:
    """Table of atomic propositions for one log."""
    def __init__(self, table: list[dict[Update | Predicate, bool]], metadata: Metadata):
        self.table = table
        self.metadata = metadata
        self.aps = set(table[0].keys()) if table else set()

    def __str__(self):
        return "|  " + "\n|  ".join(" & ".join([("" if val else "!") + str(ap) for ap, val in row.items()]) + ";" for row in self.table)

    def copy(self):
        return APTable([ { deepcopy(ap): asgn for ap, asgn in d.items() } for d in self.table ], self.metadata)

    @classmethod
    def from_log(cls, log, metadata: Metadata, self_inputs_only: bool = False):
        """Build AP table for a given trace and metadata.

            Handles scalar, tuple, and boolean variables:
            - Scalar variables: processed with Update APs
            - Tuple variables: expanded to element keys, predicates compare reconstructed tuples
            - Boolean variables: added as simple BooleanAP (true/false based on value)

            If self_inputs_only=True, only creates self-updates (playerX <- f(playerX)),
            skipping cross-updates (playerX <- f(playerY)).
        """
        if len(log) == 0:
            return APTable([], metadata)

        ap_table = []
        # NOTE: need to create id functions for all types not just ints
        idf_int = Function("", "int->int", lambda x: x)

        # Detect boolean variables from log data if not already in metadata
        # Boolean variables have Python bool values (True/False)
        detected_booleans = set()
        if not metadata.boolean_vars:
            # Auto-detect from first log entry
            for key, value in log[0].items():
                if is_boolean_value(value):
                    detected_booleans.add(key)
        else:
            detected_booleans = metadata.boolean_vars

        # Expand tuples in log entries if we have tuple_vars
        has_tuples = bool(metadata.tuple_vars)
        if has_tuples:
            log = [expand_tuple_keys(entry) for entry in log]

        for i, entry in enumerate(log):

            table_entry = {}

            # Add boolean variables as simple APs (not Updates)
            for bool_var in detected_booleans:
                if bool_var in entry:
                    ap = BooleanAP(bool_var)
                    table_entry[ap] = bool(entry[bool_var])  # True if value is True/truthy

            # Process non-tuple, non-boolean variables for updates (element-level)
            for var in metadata.vars:
                # Skip tuple-type variables for function updates (they're not directly updated)
                if "tuple" in var.type:
                    continue
                # Skip boolean variables (handled above)
                if var.name in detected_booleans:
                    continue
                if "const" not in var.type:
                    # Identity APs and simple updates for variable
                    # for var2 in metadata.vars:
                    #     if "tuple" in var2.type:
                    #         continue
                    #     if var.type == var2.type:  # same type
                    ap = Update(var, idf_int, (var,))
                    next_val = log[i+1][var.name] if i + 1 < len(log) else None
                    table_entry[ap] = (next_val is not None) and entry[var.name] == next_val

            for func in metadata.functions:
                # Permutations (with repetition) of function arguments applied to all variables
                for var_comb in product(metadata.vars, repeat=len(func.input_types)):
                    # Skip tuple-type variables
                    if any("tuple" in v.type for v in var_comb):
                        continue
                    # Skip boolean variables as inputs
                    if any(v.name in detected_booleans for v in var_comb):
                        continue
                    # Check if types match
                    if [var.type for var in var_comb] == func.input_types:
                        # Create AP for each variable which the function is applied to.
                        # Check output type matches variable type
                        for var in metadata.vars:
                            if "tuple" in var.type:
                                continue
                            # Skip boolean variables as outputs
                            if var.name in detected_booleans:
                                continue
                            # If self_inputs_only, only create self-updates (input var == output var)
                            if self_inputs_only:
                                # For unary functions, input must be the same as output
                                if len(var_comb) == 1 and var_comb[0] != var:
                                    continue
                                # Skip multi-arity functions entirely in self-inputs mode
                                if len(var_comb) > 1:
                                    continue
                            if var.type == func.output_type:
                                ap = Update(var, func, var_comb)
                                next_val = log[i+1][var.name] if i + 1 < len(log) else None
                                table_entry[ap] = func.run(*[entry[v.name] for v in var_comb]) == next_val if next_val is not None else False

            for pred in metadata.predicates:
                # Permutations (with repetition) of predicate arguments applied to all variables
                for var_comb in product(metadata.vars, repeat=len(pred.input_types)):
                    # Check if types match (handle tuple types)
                    types_match = True
                    for var, inp_type in zip(var_comb, pred.input_types):
                        if var.type != inp_type:
                            types_match = False
                            break
                    if types_match:
                        ap = Predicate(pred, var_comb)
                        # Get values for predicate evaluation
                        values = []
                        for v in var_comb:
                            if "tuple" in v.type:
                                # Reconstruct tuple from expanded elements
                                if v.name in metadata.tuple_vars:
                                    arity = metadata.tuple_vars[v.name]['arity']
                                    values.append(reconstruct_tuple(entry, v.name, arity))
                            else:
                                values.append(entry[v.name])
                        table_entry[ap] = pred.run(*values)

            table_entry["END"] = (i == len(log) - 1)

            ap_table.append(table_entry)


        return APTable(ap_table, metadata)


    def check_var_updates(self) -> list[dict[Variable, set[Update]]]:
        """Check for any timestep how many updates are applied to all variables."""
        return [{v: set(ap for ap in e if isinstance(ap, Update) and e[ap] and ap.var == v) 
                for v in self.metadata.vars}
                for e in self.table]
    

    # NOTE: Currently arbitarily choses when function ranking is the same. 
    def mutex_by_ranking(self, ranking) :
        """Make AP table respect mutex of updates by generated ranking"""
        for entry in self.table:
            for var in self.metadata.vars:
                # Get functions ranked by application count for this variable
                func_ranks = ranking[var]
                # Sort ranks descending
                sorted_ranks = sorted(func_ranks.keys(), reverse=True)
                applied = False
                for rank in sorted_ranks:
                    funcs = func_ranks[rank] 
                    # NOTE: take implicit python ordering (first in set), can be determinized
                    for func, inp in funcs: 
                        ap = Update(var, func, inp)
                        if entry[ap]:
                            if not applied:
                                applied = True
                            else:
                                # Set to false to enforce mutex
                                entry[ap] = False
    

    def to_bolt(self) -> dict[str, list[int]]:
        """Convert AP table to Bolt format."""
        d = {}
        for ap in self.aps:
            for letter in self.table:
                d[str(ap)] = d.get(str(ap), []) + [1 if letter[ap] else 0]
        return d




# NOTE: Should we rank functions by application count per variable or total application count?
#  We can return either a single list ordered by global function appication or a dictionary of lists per var 
# Current Implementation: Functions rankings (frequency) is not agnostic to application. 
#  We define a unique function as not only the function but also what it is applied to.
#  [x <- f y] and [x <- f z] are different "functions"
def rankFunctions(tables: list[APTable]) -> dict[Variable, dict[int, list[tuple[Function, tuple[Variable,...]]]]]:
    """Rank functions by how many times they are applied to each variable in the AP table."""
    updates = []
    # Merge the updates of all the tables into one update list
    for table in tables:
        updates_table = table.check_var_updates()
        for i, timestep in enumerate(updates_table):
            if i >= len(updates):
                updates.append({})
            for var, asgns in timestep.items():
                updates[i][var] = updates[i].get(var, []) + list(asgns)
                # if var not in updates[i]:
                #     updates[i][var] = []
                # updates[i][var].extend(asgns)

    func_count_vars = {var: {} for var in tables[0].metadata.vars}
    for timestep in updates:
        for var, asgns in timestep.items():
            for ap in asgns:
                if isinstance(ap, Update):
                    k = ap.func, ap.inputs # <- if we want to distinguish by application 
                    func_count_vars[var][k] = func_count_vars[var].get(k, 0) + 1
                    # break
    
    var_count_funcs = {}
    for var, func_count in func_count_vars.items():
        # Each variable has a dict of dicts(2) where key of dict(2) = count and value = list of functions with that count
        count_funcs: dict[int, list[Function]] = {}
        for func, count in func_count.items():
            count_funcs[count] = count_funcs.get(count, []) + [func]
        var_count_funcs[var] = count_funcs

    return var_count_funcs 



def cleanTables(tables: list[APTable]) -> list[APTable]:
    """Remove any variables that is unused at any point in all tables or always on accross all tables."""
    aps_in_use = set()
    aps_turned_off = set()
    all_aps = set()
    for table in tables:
        for entry in table.table:
            for ap, val in entry.items():
                if val:
                    aps_in_use.add(ap)
                if not val:
                    aps_turned_off.add(ap)
                all_aps.add(ap)
                

    print(f"All APs: {[str(ap) for ap in all_aps]}")
    print(f"APs in use: {[str(ap) for ap in aps_in_use]}")
    aps_not_in_use = all_aps - aps_in_use
    aps_never_removed = all_aps - aps_turned_off
    for table in tables:
        for entry in table.table:
            for ap in aps_not_in_use.union(aps_never_removed):
                # print(f"Removing unused AP {ap} from table.")
                del entry[ap]
        table.aps = all_aps - aps_not_in_use - aps_never_removed

    return tables



def writeBolt(pos_tables: list[APTable], neg_tables: list[APTable]) -> dict:
    assert pos_tables or neg_tables, "At least one of pos_tables or neg_tables must be non-empty."
    t = pos_tables[0] if pos_tables else neg_tables[0]

    # Sort APs: predicates first, then updates, then END
    # This ensures Bolt enumerates predicate-based formulas before update-based ones
    def ap_sort_key(ap):
        ap_str = str(ap)
        if ap_str == 'END':
            return (2, ap_str)  # END last
        elif ap_str.startswith('['):
            return (1, ap_str)  # Updates second
        else:
            return (0, ap_str)  # Predicates first

    sorted_aps = sorted(t.aps, key=ap_sort_key)

    # Reorder entries in all tables to match sorted AP order
    for table in pos_tables + neg_tables:
        table.aps = set(sorted_aps)  # Keep as set but use sorted order in bolt output
        for entry in table.table:
            # Reorder entry dict to match sorted_aps
            new_entry = {ap: entry[ap] for ap in sorted_aps if ap in entry}
            entry.clear()
            entry.update(new_entry)

    return {
        "positive_traces": [table.to_bolt() for table in pos_tables],
        "negative_traces": [table.to_bolt() for table in neg_tables],
        "atomic_propositions": [str(ap) for ap in sorted_aps],
        "number_atomic_propositions": len(sorted_aps),
        "number_positive_traces": len(pos_tables),
        "number_negative_traces": len(neg_tables),
        "max_length_traces": max(len(table.table) for table in pos_tables + neg_tables),
        "trace_type": "finite"
    }


def writeUpdatesTSL(tables: list[APTable], out_path: Path):
    """Write updates grouped by variable to a TSL-like file.

    Each line contains all updates for a single variable separated by ||.
    Only includes updates that actually appear in the cleaned traces.
    """
    if not tables:
        return

    aps = tables[0].aps

    # Group updates by variable
    updates_by_var: dict[str, list[str]] = {}
    for ap in aps:
        if isinstance(ap, Update):
            var_name = ap.var.name
            if var_name not in updates_by_var:
                updates_by_var[var_name] = []
            updates_by_var[var_name].append(str(ap))

    # Write to file
    with out_path.open("w", encoding="utf-8") as fh:
        for var_name in sorted(updates_by_var.keys()):
            updates = updates_by_var[var_name]
            line = " || ".join(sorted(updates))
            fh.write(f"{line}\n")

    print(f"Saved updates TSL to: {out_path}")




def check_empty(trace_file: Path) -> bool:
    """Check if the trace file is empty."""
    with trace_file.open("r", encoding="utf-8") as fh:
        return not any(line.strip() for line in fh)
    

def main():
    args = parse_args()
    print(f"Processing traces from: {args.traces}")
    print(f"Using metadata from: {args.meta}")

    self_inputs_only = getattr(args, 'self_inputs_only', False)
    if self_inputs_only:
        print("Mode: self-inputs-only (skipping cross-updates)")

    metadata = Metadata.import_metadata(args.meta)
    print("Metadata loaded successfully:")
    print(metadata)

    pos = args.pos if args.pos else args.traces / "pos"
    neg = args.neg if args.neg else args.traces / "neg"

    pos_tables = {}
    for trace_file in pos.glob("*.jsonl"):
        if check_empty(trace_file):
            print(f"Skipping empty trace file: {trace_file}")
            continue
        print(f"Processing positive trace: {trace_file}")
        # pos_tables[trace_file.stem] = generate_ap_tables(trace_file, metadata)
        log = [json.loads(line) for line in trace_file.open("r", encoding="utf-8") if line.strip()]
        pos_tables[trace_file.stem] = APTable.from_log(log, metadata, self_inputs_only=self_inputs_only)
        print(pos_tables[trace_file.stem])
        print("----------------")


    neg_tables = {}
    for trace_file in neg.glob("*.jsonl"):
        if check_empty(trace_file):
            print(f"Skipping empty trace file: {trace_file}")
            continue
        print(f"Processing negative trace: {trace_file}")
        log = [json.loads(line) for line in trace_file.open("r", encoding="utf-8") if line.strip()]
        neg_tables[trace_file.stem] = APTable.from_log(log, metadata, self_inputs_only=self_inputs_only)
        print(neg_tables[trace_file.stem])
        # neg_tables[trace_file.stem] = generate_ap_tables(trace_file, metadata)
        print("----------------")

    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

    tables = list(pos_tables.values()) + list(neg_tables.values())

    func_ranks = rankFunctions(tables)
    for var, funcs in func_ranks.items():
        print(f" FuncRank {var}: {[(i, [(str(f), [str(i) for i in inp]) for f, inp in funcs]) for i, funcs in funcs.items()]}")

    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

    for table_dict in [pos_tables, neg_tables]:
        for table_name, table in table_dict.items():
            table.mutex_by_ranking(func_ranks)
            # print(table)
            num_updates = table.check_var_updates()
            print(f"Number of updates at each time step for {table_name} after mutex:")
            for i, updates in enumerate(num_updates):
                print(f" {i}: {[(str(u), [str(f) for f in v]) for u, v in updates.items()]}")
            print("----------------")
    
    clean = cleanTables(tables)
    pos_clean, neg_clean = clean[:len(pos_tables)], clean[len(pos_tables):]

    bolt_entry = writeBolt(pos_clean, neg_clean)
    # print("BOLT entry:")
    if args.out:
        out_file = args.out / f"bolt.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(bolt_entry, fh)
        print(f"Saved BOLT entry to: {out_file}")

        # Write updates TSL file
        updates_file = args.out / "updates.tsl"
        writeUpdatesTSL(clean, updates_file)
    else:
        print(json.dumps(bolt_entry))
    print("----------------")

    # exit(0)

    # combs = {}
    # for name, tables in zip(["pos_combinations", "neg_combinations"], [pos_tables, neg_tables]):
    #     combs[name] = cartesian_product_tables(tables)
    #     print(f"Generated {len(combs[name])} combinations of {name}:")
    #     # for combo in combs[name]:
    #     #     for table in combo:
    #     #         print(table)
    #             # print(f"Number of updates at each time step:")
    #             # for i, updates in enumerate(table.check_updates()):
    #             #     print(f" {i}: {updates}")
    #         # print("----")

    # # produce explicit references to pos/neg combinations and take their cartesian product
    # pos_combinations = combs.get("pos_combinations", [])
    # neg_combinations = combs.get("neg_combinations", [])

    # # pair each positive combination with each negative combination
    # pos_neg_product = [(deepcopy(p), deepcopy(n)) for p in pos_combinations for n in neg_combinations]

    # exit(1)
    # # print("\n\n\n")
    # for i, (pos_combo, neg_combo) in enumerate(pos_neg_product):
    #     print(f"Combination {i+1}:")
    #     print("Positive combination:")
    #     for table in pos_combo:
    #         print(table)
    #     print("Negative combination:")
    #     for table in neg_combo:
    #         print(table)

    #     cleaned_tables = cleanup_ap_tables(pos_combo + neg_combo)
    #     pos_combo_cleaned, neg_combo_cleaned = cleaned_tables[:len(pos_combo)], cleaned_tables[len(pos_combo):]
        
    #     bolt_entry = write_bolt_dict(pos_combo_cleaned, neg_combo_cleaned)
    #     # print("BOLT entry:")
    #     if args.out:
    #         out_file = args.out / f"combination_{i+1}.json"
    #         with out_file.open("w", encoding="utf-8") as fh:
    #             json.dump(bolt_entry, fh)
    #         print(f"Saved BOLT entry to: {out_file}")
    #     else:
    #         print(json.dumps(bolt_entry))
    #     print("----------------")

    # print(f"Generated {len(pos_neg_product)} positive-negative combination pairs.")

if __name__ == "__main__":
    main()
                    
                



# def cartesian_product_tables(table_dict: dict[str, list[APTable]]) -> list[list[APTable]]:
#     """
#     Return all combinations of picking one APTable from each entry of table_dict.
#     The order of each combination follows the iteration order of table_dict.keys().
#     """
#     lists = list(table_dict.values())
#     return [list(combo) for combo in product(*lists)]

# def generate_ap_tables(trace_file: Path, metadata: Metadata) -> list[APTable]:
#     with trace_file.open("r", encoding="utf-8") as fh:
#         log = [json.loads(line) for line in fh if line.strip()]
#         good_tables = []
#         while good_tables == []:
#             table = APTable.from_log(log, metadata)
#             # table = build_ap_table(log, metadata)
#             print(f"AP table for {trace_file.name}:")
#             print(table)

#             var_updates = table.check_var_updates()
#             print(f"Number of updates at each time step for {trace_file.name}:")
#             for i, updates in enumerate(var_updates):
#                 print(f" {i}: {[(str(u), len(v)) for u, v in updates.items()]}")
#                 # print(f" {i}: {[(str(u), str(v)) for u, v in updates.items()]}")

#             func_ranks = table.rank_functions(var_updates)
#             for var, funcs in func_ranks.items():
#                 print(f" FuncRank {var}: {[(i, [(str(f), [str(i) for i in inp]) for f, inp in funcs]) for i, funcs in funcs.items()]}")
            
#             table.mutex_by_ranking(var_updates, func_ranks)
#             print("AP table after enforcing mutex by ranking:")
#             print(table)

#             num_updates = table.check_var_updates()
#             print(f"Number of updates at each time step for {trace_file.name} after mutex:")
#             # for i, updates in enumerate(num_updates):
#             #     print(f" {i}: {[(str(u), len(v)) for u, v in updates.items()]}")
#             for i, updates in enumerate(num_updates):
#                 print(f" {i}: {[(str(u), [str(f) for f in v]) for u, v in updates.items()]}")
#                 # print(f" {i}: {[(str(u), str(v)) for u, v in updates.items()]}")

#             exit(1)

#             # NOTE: Function composition turned off for now
#             # if any(any(update == 0 for update in updates.values()) for updates in num_updates[:-1]):
#             #     print("Detected table entry with no updates. Composing functions...")
#             #     metadata = metadata.compose_functions()
#             #     print("Generated new functions:")
#             #     print(metadata)
#             # NOTE: was elif here
#             if any(any(len(update) > 1 for update in updates.values()) for updates in var_updates[:-1]):
#                 print("Detected table entry with multiple updates. Splitting mining traces...")
#                 tables = table.split_tables2()
#                 print(f"Split into {len(tables)} tables:")
#                 # for t in tables:
#                 #     print("-------------------")
#                 #     print(t)
#                 #     for i, updates in enumerate(t.check_updates()):
#                 #         print(f" {i}: {updates}")
#                 good_tables.extend(tables)
#                 # print(f"TOTAL GOOD TABLES SO FAR: {len(good_tables)}")
#                 # exit(1)
#             else :
#                 print("All entries are updated exactly once.")
#                 good_tables.append(table)
        
#     return good_tables




    # def split_tables(self, i=0) -> list[Self]:
    #     """Split AP table into multiple tables if multiple updates are applied to a term at the same time step.
    #         e.g. Suppose at time step 0, we have:
    #         APTable([{"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": True, "rightMost ball": False, "leftMost ball": False, "END": False}
    #         Then we would need to create two tables:
    #         [
    #             APTable([{"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": False, "END": False},
    #             APTable([{"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False}
    #         ]
    #     """

    #     # print(f"split_tables: i={i}")
    #     if i == len(self.table) - 1:
    #         return [self]
        
    #     updates_at_time_step = {var: [ap for ap in self.table[i] 
    #                                     if (isinstance(ap, Update) or isinstance(ap, UpdateF)) 
    #                                         and ap.var == var and self.table[i][ap]] 
    #                                     for var in self.metadata.vars.keys()}
    #     out = []
    #     iterated = False
    #     new_tables = []
    #     for var, asgns in updates_at_time_step.items():
    #         print(f"*** {var} : {[str(k) for k in updates_at_time_step[var]]}")
    #         if len(asgns) > 1:
    #             iterated = True
    #             for asgn in asgns:
    #                 print(f"{i} asgn={asgn}")
    #                 new_table = self.copy()
    #                 for asgn2 in asgns:
    #                     if asgn != asgn2:
    #                         new_table.table[i][asgn2] = False
    #                 print(new_table)
    #                 exit(1)
    #                 print("------")
    #                 out.extend(new_table.split_tables(i+1))
        
    #     if not iterated:
    #         out.extend(self.split_tables(i+1))
    #     return out
    

    # def split_tables2(self, i=0) -> list[Self]:
    #     """
    #     Split AP table into multiple tables if multiple updates are applied to a term at the same time step.
    #     e.g. Suppose at time step 0, we have:
    #     APTable([{"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": True, "rightMost ball": False, "leftMost ball": False, "END": False}
    #     Then we would need to create two tables:
    #     [
    #         APTable([{"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": False, "END": False},
    #         APTable([{"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False}
    #     ]
        
    #     BFS approach: split AP table by generating all permutations of updates at each timestep.
    #     For each timestep, if any variable has multiple true updates, generate all permutations
    #     where each variable gets exactly one true update. Then recursively process the next timestep.
    #     """
    #     if i == len(self.table) - 1:
    #         return [self]
        
    #     # Find all variables with multiple true updates at timestep i
    #     updates_at_time_step = {var: asgn for var, asgn in self.check_var_updates()[i].items() if len(asgn) > 1}
        
    #     if not updates_at_time_step:
    #         # No conflicts at this timestep, move to next
    #         return self.split_tables2(i+1)
        
    #     # Generate all permutations of choices for variables with multiple updates
    #     vars_with_conflicts = list(updates_at_time_step.keys())
    #     update_choices = [updates_at_time_step[var] for var in vars_with_conflicts]
        
    #     all_tables = []
    #     for choice in product(*update_choices):
    #         # choice is a tuple of (update1, update2, ...) for each conflicting variable
    #         new_table = self.copy()
            
    #         # For each variable with conflicts, set all updates to False except the chosen one
    #         for var, chosen_update in zip(vars_with_conflicts, choice):
    #             for ap in updates_at_time_step[var]:
    #                 if ap != chosen_update:
    #                     new_table.table[i][ap] = False
            
    #         # Recursively process the next timestep
    #         all_tables.extend(new_table.split_tables2(i+1))
        
    #     return all_tables



    # for trace_file in args.traces.glob("*.jsonl"):
    #     print(f"Processing trace: {trace_file}")
    #     trace = Trace.from_jsonl(trace_file, metadata)
    #     out_file = trace_file.with_suffix(".trace")
    #     trace.to_trace(out_file)



# Table = list[dict[Update | UpdateF | Predicate, bool]]

# def build_ap_table(log, metadata) -> Table:
#     """Build AP table for a given trace and metadata.
    
#         e.g. Suppose log:
#         {"ball": 1}
#         {"ball": 2}
#         {"ball": 1}
#         {"ball": 0}
#         {"ball": -1}
#         {"ball": -2}

#         And metadata:
#         VARS = {
#             "ball": int
#         }
#         FUNCTIONS = {
#             "moveRight": ("int->int", lambda ball: ball + 1),
#             "moveLeft": ("int->int", lambda ball: ball - 1),
#         }
#         PREDICATES = {
#             "rightMost": ("int->bool", lambda ball: ball == -2),
#             "leftMost": ("int->bool", lambda ball: ball == 2),
#         }

#         Then AP table would be:
#         [
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": False, "END": False},
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": True,  "leftMost ball": False, "END": False},
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False},
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False},
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False},
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": True,  "END": True}
#         ]
#     """
#     ap_table = []

#     for i, entry in enumerate(log):

#         table_entry = {}
#         for var_name, _ in metadata["vars"].items():
#             # Identity APs and simple updates for variable
#             for var2_name, _ in metadata["vars"].items():
#                 if metadata["vars"][var_name] == metadata["vars"][var2_name]: # same type
#                     ap = Update(var_name, var2_name)
#                     next_val = log[i+1][var_name] if i + 1 < len(log) else None
#                     table_entry[ap] = entry[var2_name] == next_val if next_val is not None else False
    
#         for _, func in metadata["functions"].items():
#             # Permutations (with repetition) of function arguments applied to all variables
#             # print(f"[biuld_ap_table] combs={list(product(metadata['vars'].keys(), repeat=len(func.input_types)))}")
#             for var_comb in product(metadata["vars"].keys(), repeat=len(func.input_types)):
#                 # Check if types match
#                 if [metadata["vars"][var] for var in var_comb] == func.input_types:
#                     # print(f"[ap_table] func={func}, var_comb={var_comb}")
#                     # print(f"[ap_table] var_comb={var_comb}")

#                     # Create AP for each variable which the function is applied to.
#                     # Check output type matches variable type
#                     for var in metadata["vars"].keys():
#                         if metadata["vars"][var] == func.output_type:
#                             ap = UpdateF(var, func, var_comb)
#                             next_val = log[i+1][var] if i + 1 < len(log) else None
#                             table_entry[ap] = func.run(*[entry[v] for v in var_comb]) == next_val if next_val is not None else False
            
#         for _, pred in metadata["predicates"].items():
#             # Permutations (with repetition) of predicate arguments applied to all variables
#             for var_comb in product(metadata["vars"].keys(), repeat=len(pred.input_types)):
#                 # Check if types match
#                 if [metadata["vars"][var] for var in var_comb] == pred.input_types:
#                     ap = Predicate(pred, var_comb)
#                     table_entry[ap] = pred.run(*[entry[v] for v in var_comb])
        
#         table_entry["END"] = (i == len(log) - 1)

#         ap_table.append(table_entry)
 
    
#     return ap_table


# def check_updates(ap_table: Table, vrs: dict[str, str]) -> list[dict[str, int]]:
#     """Check for any point how many updates are applied to a term."""
#     num_updates = []
#     for entry in ap_table:
#         updates_at_time_step = {var: sum(entry[ap] for ap in entry if (isinstance(ap, Update) or isinstance(ap, UpdateF)) and ap.var == var) for var in vrs.keys()}
#         num_updates.append(updates_at_time_step)
#     return num_updates

# def copy_table(table: Table) -> Table:
#     return [{ deepcopy(ap): asgn for ap, asgn in d.items() } for d in table]


# # class Tree :
# #     def __init__(self):



# def split_tables(ap_table: Table, vrs: dict[str, str], i=0) -> list[Table]:
#     """Split AP table into multiple tables if multiple updates are applied to a term at the same time step.
#         e.g. Suppose at time step 0, we have:
#         {"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": True, "rightMost ball": False, "leftMost ball": False, "END": False}
#         Then we would need to create two tables:
#         [
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": True,  "[ball <- moveLeft ball]": False, "rightMost ball": False, "leftMost ball": False, "END": False},
#             {"[ball <- ball]": False, "[ball <- moveRight ball]": False, "[ball <- moveLeft ball]": True,  "rightMost ball": False, "leftMost ball": False, "END": False}
#     ]
#     """

#     print(f"split_tables: i={i}")

#     if i == len(ap_table) - 1:
#         return [ap_table]
    
#     updates_at_time_step = {var: [ap for ap in ap_table[i] 
#                                     if (isinstance(ap, Update) or isinstance(ap, UpdateF)) 
#                                         and ap.var == var and ap_table[i][ap]] 
#                                     for var in vrs.keys()}
    
#     out = []
#     splitted = False
#     for var, asgns in updates_at_time_step.items():
#         print(var, [str(k) for k in updates_at_time_step[var]])
#         if len(asgns) > 1:
#             splitted = True
#             for asgn in asgns:
#                 print(f"asgn={asgn}")
#                 new_table = copy_table(ap_table)
#                 for asgn2 in asgns:
#                     if asgn != asgn2:
#                         new_table[i][asgn] = False
#                 print_table(new_table)
#                 # print(f"new table: {new_table}")
#                 out.extend(split_tables(new_table, vrs, i+1))
    
#     if not splitted :
#         out.extend(split_tables(ap_table, vrs, i+1))
    
#     return out

        



# def print_table(ap_table: Table):
#     print("  " + "\n  ".join(str([("" if val else "!") + str(ap) for ap, val in row.items()])[1:-1] for row in ap_table))


# def compose_metadata_functions(metadata):
#     """Compose functions in metadata to create new functions."""
#     new_functions = {}
#     for f1 in metadata["functions"].values():
#         for f2 in metadata["functions"].values():
#             for i, ty in enumerate(f2.input_types):

#                 # NOTE: Only composing currently on first arg. Should be extended to multiple args
#                 if f1.output_type == ty:  # Check if output of f1 matches input of f2
#                     composed_name = f"{f2.name}({i}{f1.name})"
#                     composed_arg_types = "->".join(f2.input_types[:i] + f1.input_types + f2.input_types[i+1:] + [f2.output_type])
#                     composed_impl = lambda *args, f1=f1, f2=f2: f2.run(f1.run(*args[:len(f1.input_types)]), *args[len(f1.input_types):])
#                     new_functions[composed_name] = (composed_arg_types, composed_impl)
    
#     metadata["functions"].update({name: Function(name, *details) for name, details in new_functions.items()})
#     return metadata