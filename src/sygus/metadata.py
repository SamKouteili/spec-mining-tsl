import os
import json
import argparse
from typing import Any
from groupings import find_constant_variables, find_boolean_variables, expand_tuple_keys, is_tuple_value, get_tuple_structure

OP_PREFIX = {
    "+": "add",
    "*": "mul",
    "-": "sub",
}

# Issy function format uses these operators
ISSY_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
}


def _expr_to_issy(expr: Any, arg_names: list[str] = None) -> str:
    """
    Convert a parsed SyGuS expression to Issy format.

    Issy format uses:
    - Primitive operations: add, sub, mul
    - Integer constants as nullary functions: i1(), i2(), i0(), etc.
    - Nested expressions with parentheses

    Examples:
        (+ x 1)     → add x i1()
        (- x 1)     → sub x i1()
        (- 1 x)     → sub i1() x
        x           → x (identity, just the variable placeholder)
        (+ (+ x x) 1) → add (add x x) i1()

    Args:
        expr: Parsed S-expression (list, str, or int)
        arg_names: List of argument names in the function signature (e.g., ["x", "y"])

    Returns:
        Issy format string with placeholders like {0}, {1} for arguments
    """
    if arg_names is None:
        arg_names = ["x"]

    if isinstance(expr, str):
        # Variable reference - convert to placeholder
        if expr in arg_names:
            idx = arg_names.index(expr)
            return f"{{{idx}}}"  # {0}, {1}, etc.
        return expr  # Unknown variable, keep as-is

    if isinstance(expr, int):
        # Integer constant → iN()
        if expr < 0:
            # Negative numbers: i-1() doesn't work, use sub i0() i1() pattern
            return f"(sub i0() i{abs(expr)}())"
        return f"i{expr}()"  # No spaces inside: i1(), i2(), etc.

    if isinstance(expr, list) and len(expr) >= 2:
        op = expr[0]
        if op in ISSY_OPS:
            issy_op = ISSY_OPS[op]
            # Convert operands recursively
            operands = [_expr_to_issy(e, arg_names) for e in expr[1:]]

            # Binary operations
            if len(operands) == 2:
                left, right = operands
                # If operands are compound (contain spaces or parens), they need parens
                # But simple atoms don't need parens
                return f"{issy_op} {left} {right}"
            elif len(operands) == 1:
                # Unary minus: (- x) means negation
                return f"sub i0() {operands[0]}"

        # Unknown operator, return as-is with conversion attempt
        return f"({op} {' '.join(_expr_to_issy(e, arg_names) for e in expr[1:])})"

    # Fallback
    return str(expr)


def _wrap_compound_issy(issy_str: str) -> str:
    """
    Wrap a compound Issy expression in parentheses if needed.
    A compound expression has spaces and isn't already wrapped.
    """
    if ' ' in issy_str and not (issy_str.startswith('(') and issy_str.endswith(')')):
        return f"({issy_str})"
    return issy_str


def _finalize_issy_expr(raw_issy: str) -> str:
    """
    Post-process the raw Issy expression to properly parenthesize nested operations.

    Example: "add add x x i1()" needs to become "add (add x x) i1()"

    The raw expression from _expr_to_issy is already well-formed with proper
    operator-argument structure. We just need to parenthesize compound arguments.
    """
    tokens = _tokenize_issy(raw_issy)
    if not tokens:
        return raw_issy
    result, _ = _parse_one_issy_arg(tokens)
    return result


def _tokenize_issy(s: str) -> list[str]:
    """
    Tokenize an Issy expression.

    Handles:
    - Operators: add, sub, mul
    - Integer constants: i0(), i1(), i2(), etc. (kept as single tokens)
    - Placeholders: {0}, {1}, etc.
    - Parentheses for grouping
    """
    tokens = []
    i = 0
    while i < len(s):
        char = s[i]

        # Skip whitespace
        if char.isspace():
            i += 1
            continue

        # Parentheses (but not part of iN())
        if char == '(':
            # Check if this is a grouping paren or part of iN()
            # iN() looks like: we just saw iN and now see ()
            tokens.append('(')
            i += 1
            continue

        if char == ')':
            tokens.append(')')
            i += 1
            continue

        # Placeholder like {0}, {1}
        if char == '{':
            end = s.find('}', i)
            if end != -1:
                tokens.append(s[i:end+1])
                i = end + 1
                continue

        # Check for iN() pattern (integer constant)
        if char == 'i' and i + 1 < len(s) and s[i+1].isdigit():
            # Find the end of the number
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            # Check for ()
            if j < len(s) - 1 and s[j:j+2] == '()':
                tokens.append(s[i:j+2])  # e.g., "i1()"
                i = j + 2
                continue

        # Regular identifier (add, sub, mul, etc.)
        if char.isalpha() or char == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue

        # Skip unknown characters
        i += 1

    return tokens


def _parse_one_issy_arg(tokens: list[str], needs_parens: bool = False) -> tuple[str, list[str]]:
    """
    Parse one argument from the token stream.
    Returns (parsed_arg, remaining_tokens).

    Arguments can be:
    - Simple atoms: {0}, i1(), i2()
    - Operators followed by two arguments: add x y
    - Parenthesized expressions: (add x y)

    Args:
        tokens: Token stream
        needs_parens: If True, compound expressions will be wrapped in parens
    """
    if not tokens:
        return "", []

    token = tokens[0]

    # Parenthesized sub-expression
    if token == '(':
        depth = 1
        i = 1
        while i < len(tokens) and depth > 0:
            if tokens[i] == '(':
                depth += 1
            elif tokens[i] == ')':
                depth -= 1
            i += 1
        inner = tokens[1:i-1]
        if inner:
            inner_str, _ = _parse_one_issy_arg(inner, needs_parens=False)
        else:
            inner_str = ""
        return f"({inner_str})", tokens[i:]

    # Operator starts a compound expression - parse two arguments
    if token in ['add', 'sub', 'mul']:
        rest = tokens[1:]
        # Arguments to this operator need parens if they're compound
        arg1, remaining = _parse_one_issy_arg(rest, needs_parens=True)
        arg2, final_remaining = _parse_one_issy_arg(remaining, needs_parens=True)
        result = f"{token} {arg1} {arg2}"
        # Wrap in parens if this is an argument to another operator
        if needs_parens:
            return f"({result})", final_remaining
        return result, final_remaining

    # Simple atom (placeholder {0}, constant i1(), etc.)
    return token, tokens[1:]

# Map algebraic expressions to semantic operation names
def _get_semantic_op_name(expr: Any) -> str:
    """
    Convert an algebraic expression to a semantic operation name.
    The suffix indicates the constant value in the operation:
    - (+ x 1) → inc1 (increment by 1)
    - (+ x 2) → inc2 (increment by 2)
    - (- x 1) → dec1 (decrement by 1)
    - x → id (identity)
    - (- 1 x) → flip1 (1 minus x)
    - (- 2 x) → flip2 (2 minus x)
    """
    if isinstance(expr, str):
        # Variable reference = identity
        return "id"
    if isinstance(expr, int):
        return f"const{expr}"
    if isinstance(expr, list) and len(expr) >= 2:
        op = expr[0]
        if op == "+" and len(expr) == 3:
            # Check for (+ x N) or (+ N x)
            if isinstance(expr[2], int) and isinstance(expr[1], str):
                return f"inc{expr[2]}"
            if isinstance(expr[1], int) and isinstance(expr[2], str):
                return f"inc{expr[1]}"
            return "add"
        if op == "-" and len(expr) == 3:
            # Check for (- x N) → decN
            if isinstance(expr[2], int) and isinstance(expr[1], str):
                return f"dec{expr[2]}"
            # Check for (- N x) → flipN
            if isinstance(expr[1], int) and isinstance(expr[2], str):
                return f"flip{expr[1]}"
            return "sub"
    return "fn"

def clean_line(raw):
    return (
        raw.replace("\ufeff", "")
           .replace("\xa0", " ")
           .strip()
    )

def extract_vars(file_path, constants, expand_tuples=True) -> dict[str, str]:
    """
    Return a dictionary of variables in the logs and their types.
    If expand_tuples=True, tuple values are expanded to element variables.
    """
    d = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            clean = clean_line(raw)
            if not clean:
                continue
            try:
                obj = json.loads(clean)
            except Exception:
                continue
            if isinstance(obj, dict):
                if expand_tuples:
                    obj = expand_tuple_keys(obj)
                for k in obj.keys():
                    d[k] = ("const " if k in constants else "") + type(obj[k]).__name__
            return d
    return d


def get_tuple_vars(file_path) -> dict[str, int]:
    """
    Get tuple variables and their arities from the first line of a trace file.
    Returns a dict mapping tuple variable names to their element count.
    """
    return get_tuple_structure(file_path)


def _extract_define_block(raw: str) -> str | None:
    """
    Extract the (define-fun ...) block from a raw SyGuS snippet.
    """
    start = raw.find("(define-fun")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(raw)):
        char = raw[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _tokenize_sygus(expr: str) -> list[str]:
    tokens = []
    current = []
    for char in expr:
        if char in "()":
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _parse_tokens(tokens: list[str]) -> Any:
    if not tokens:
        raise ValueError("Unexpected end of tokens")
    token = tokens.pop(0)
    if token == "(":
        result = []
        while tokens and tokens[0] != ")":
            result.append(_parse_tokens(tokens))
        if not tokens:
            raise ValueError("Unbalanced parentheses in SyGuS snippet")
        tokens.pop(0)
        return result
    if token == ")":
        raise ValueError("Unexpected closing parenthesis")
    try:
        return int(token)
    except ValueError:
        return token


def _parse_define_fun(block: str) -> list[Any]:
    tokens = _tokenize_sygus(block)
    parsed = _parse_tokens(tokens)
    return parsed


def _capitalize(token: str) -> str:
    if not token:
        return token
    return token[0].upper() + token[1:]


def _operand_token(node: Any) -> str:
    if isinstance(node, list):
        return _build_function_name(node, top=False)
    if isinstance(node, str):
        return node.capitalize()
    if isinstance(node, int):
        return str(node)
    return "Val"


def _build_function_name(node: Any, top: bool = True) -> str:
    if isinstance(node, list):
        op = node[0]
        prefix = OP_PREFIX.get(op, "fn")
        suffix = "".join(_operand_token(child) for child in node[1:])
        name = prefix + suffix
        return name if top else _capitalize(name)
    if isinstance(node, str):
        token = node.capitalize()
        return f"id{token}" if top else token
    if isinstance(node, int):
        token = str(node)
        return f"const{token}" if top else token
    return "fn"


def _build_python_expression(node: Any) -> str:
    if isinstance(node, list):
        op = node[0]
        if op not in {"+", "*", "-"}:
            raise ValueError(f"Unsupported operator {op}")
        exprs = [_build_python_expression(child) for child in node[1:]]
        if op == "+":
            joiner = " + "
            return "(" + joiner.join(exprs) + ")"
        if op == "*":
            joiner = " * "
            return "(" + joiner.join(exprs) + ")"
        # subtraction
        if len(exprs) == 1:
            return f"(-{exprs[0]})"
        head, *rest = exprs
        return "(" + " - ".join([head] + rest) + ")"
    if isinstance(node, int):
        return str(node)
    if isinstance(node, str):
        return node
    raise ValueError(f"Unsupported node type: {type(node)}")


def _build_lambda(arg_names: list[str], body: str):
    args = ", ".join(arg_names) if arg_names else "*_"
    src = f"lambda {args}: {body}"
    return eval(src, {"__builtins__": {}})


def _render_type(arity: int) -> str:
    if arity == 0:
        return "->int"
    domain = ",".join(["int"] * arity)
    return f"{domain}->int"


def _normalize_expr(node: Any, mapping: dict[str, str]):
    if isinstance(node, list):
        return (node[0], tuple(_normalize_expr(child, mapping) for child in node[1:]))
    if isinstance(node, str):
        return mapping.get(node, node)
    return node


def _canonical_key(expr: Any, arg_names: list[str], signature: str):
    mapping = {name: f"ARG{i}" for i, name in enumerate(arg_names)}
    normalized = _normalize_expr(expr, mapping)
    return (signature, normalized)


def _extract_element_index(var_name: str) -> int | None:
    """
    Extract the element index from a variable name like 'player[0]' -> 0.
    Returns None if no index found.
    """
    import re
    match = re.search(r'\[(\d+)\]$', var_name)
    if match:
        return int(match.group(1))
    return None


def extract_function_metadata(functions_path: str):
    """
    Convert solver output functions into callable Python metadata.
    Creates element-aware function names like inc0, dec1, id0.
    """
    with open(functions_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = payload.get("functions", [])
    assignments = payload.get("assignments", {})
    assert isinstance(entries, list), "Expected function entries to be a list"

    # First pass: parse all functions and get their expressions
    parsed_functions = []  # List of (func_id, expr, arg_names, python_expr)
    for func_id, raw in enumerate(entries):
        if not isinstance(raw, str):
            parsed_functions.append((func_id, None, [], "x"))
            continue

        # Handle IDENTITY marker - skip it, identity is handled implicitly by log2tslf.py
        if raw == "IDENTITY":
            parsed_functions.append((func_id, None, [], "x"))  # Mark as None to skip later
            continue

        block = _extract_define_block(raw)
        if not block:
            parsed_functions.append((func_id, None, [], "x"))
            continue
        try:
            parsed = _parse_define_fun(block)
        except ValueError:
            parsed_functions.append((func_id, None, [], "x"))
            continue
        if not parsed or parsed[0] != "define-fun":
            parsed_functions.append((func_id, None, [], "x"))
            continue
        args_section = parsed[2] if len(parsed) > 2 else []
        arg_names = [arg[0] for arg in args_section if isinstance(arg, list) and arg]
        expr = parsed[4] if len(parsed) > 4 else None
        if expr is None:
            parsed_functions.append((func_id, None, [], "x"))
            continue
        try:
            python_expr = _build_python_expression(expr)
        except ValueError:
            python_expr = "x"
        parsed_functions.append((func_id, expr, arg_names, python_expr))

    # Second pass: determine which elements each function is applied to
    # and create element-specific function names
    func_element_map = {}  # (func_id, element_idx) -> set of input_vars
    for key, assignment in assignments.items():
        # Key format: "time_N__var[M]"
        parts = key.split("__")
        if len(parts) != 2:
            continue
        var_part = parts[1]  # e.g., "player[0]"
        element_idx = _extract_element_index(var_part)
        if element_idx is None:
            element_idx = 0  # Default for non-tuple vars
        func_id = assignment.get("function_id", 0)
        record = assignment.get("record", {})
        input_vars = record.get("_input_vars", [])

        map_key = (func_id, element_idx)
        if map_key not in func_element_map:
            func_element_map[map_key] = set()
        func_element_map[map_key].update(input_vars)

    # Third pass: create metadata with semantic names (no element suffix)
    # Deduplicate by (semantic_op, python_expr) so we only have one inc, one id, etc.
    metadata: dict[str, tuple] = {}
    specs: dict[str, dict] = {}
    seen_ops: dict[tuple, str] = {}  # (semantic_op, python_expr) -> name

    for (func_id, element_idx), input_vars in func_element_map.items():
        if func_id >= len(parsed_functions):
            continue
        _, expr, arg_names, python_expr = parsed_functions[func_id]

        # Skip identity functions - they're handled implicitly by log2tslf.py
        if expr is None or (isinstance(expr, str) and python_expr == "x"):
            continue

        # Get semantic operation name
        semantic_op = _get_semantic_op_name(expr)

        # Skip identity (in case _get_semantic_op_name returns "id")
        if semantic_op == "id":
            continue

        # Deduplicate by (semantic_op, python_expr)
        dedup_key = (semantic_op, python_expr)
        if dedup_key in seen_ops:
            continue  # Already have this function

        # Use just the semantic operation name (no element suffix)
        func_name = semantic_op

        # Handle duplicates (different implementations with same semantic name)
        unique_name = func_name
        counter = 2
        while unique_name in metadata:
            unique_name = f"{func_name}_{counter}"
            counter += 1

        seen_ops[dedup_key] = unique_name

        arity = len(arg_names) if arg_names else 1
        sig = _render_type(arity)

        try:
            impl = _build_lambda(arg_names if arg_names else ["x"], python_expr)
        except Exception:
            impl = lambda x: x

        # Compute Issy template for this function
        raw_issy = _expr_to_issy(expr, arg_names if arg_names else ["x"])
        issy_template = _finalize_issy_expr(raw_issy)

        metadata[unique_name] = (sig, impl)
        specs[unique_name] = {
            "type": sig,
            "args": arg_names if arg_names else ["x"],
            "expr": python_expr,
            "key": (sig, semantic_op, python_expr),
            "base_name": func_name,
            "semantic_op": semantic_op,
            "issy_template": issy_template,  # e.g., "add {0} i1()" for inc1
        }

    return metadata, specs

def _format_vars_block(variables: dict[str, str]) -> list[str]:
    lines = ["VARS = {"]
    for key in sorted(variables.keys()):
        value = variables[key]
        lines.append(f'    "{key}": "{value}",')
    if len(lines) == 1:
        lines.append("    # No variables detected")
    lines.append("}")
    return lines


def _format_functions_block(function_specs: dict[str, dict[str, Any]]) -> list[str]:
    """
    Format the FUNCTIONS block for metadata.py.

    Each function entry is a tuple: (type_sig, lambda, issy_template)
    - type_sig: e.g., "int->int"
    - lambda: Python callable implementing the function
    - issy_template: Issy format string with placeholders, e.g., "add {0} i1()"
    """
    lines = ["FUNCTIONS = {"]
    for name in sorted(function_specs.keys()):
        spec = function_specs[name]
        args = spec.get("args", [])
        lambda_args = ", ".join(args) if args else "_"
        expr = spec.get("expr", "0")
        type_sig = spec.get("type", "int->int")
        issy_template = spec.get("issy_template", "{0}")  # Default to identity-like
        lines.append(
            f'    "{name}": ("{type_sig}", lambda {lambda_args}: {expr}, "{issy_template}"),'
        )
    if len(lines) == 1:
        lines.append("    # No functions synthesized")
    lines.append("}")
    return lines


def write_metadata_file(out_dir: str, variables: dict[str, str], function_specs: dict[str, dict[str, Any]],
                        tuple_vars: dict[str, int] = None, boolean_vars: set[str] = None):
    """
    Write VARS and FUNCTIONS dictionaries matching the f/ball/ball.py style.
    If tuple_vars is provided, also generates tuple equality predicates.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metadata.py")
    content_lines: list[str] = []
    content_lines.extend(_format_vars_block(variables))
    content_lines.append("")
    content_lines.extend(_format_functions_block(function_specs))
    content_lines.append("")

    # Add predicates section
    content_lines.append("PREDICATES = {")

    if tuple_vars:
        # Find the tuple arity (assume all tuples have same arity for now)
        arities = set(tuple_vars.values())
        if len(arities) == 1:
            arity = arities.pop()
            # Single generic eq predicate for comparing any two tuples
            # eq compares dynamic tuple to dynamic tuple
            content_lines.append(f'    "eq": ("tuple{arity}->tuple{arity}->bool", lambda x, y: x == y),')
            # eqC compares dynamic tuple to constant tuple
            content_lines.append(f'    "eqC": ("tuple{arity}->const tuple{arity}->bool", lambda x, y: x == y),')
        else:
            # Multiple arities - generate predicates for each
            for arity in arities:
                content_lines.append(f'    "eq{arity}": ("tuple{arity}->tuple{arity}->bool", lambda x, y: x == y),')
                content_lines.append(f'    "eqC{arity}": ("tuple{arity}->const tuple{arity}->bool", lambda x, y: x == y),')

    else:
        # Fallback to simple integer equality predicates
        content_lines.append('    "eq": ("int->int->bool", lambda x, y: x == y),')
        content_lines.append('    "eqC": ("int->const int->bool", lambda x, y: x == y),')
        content_lines.append('    "lt": ("int->int->bool", lambda x, y: x < y),')
        content_lines.append('    "ltC": ("int->const int->bool", lambda x, y: x < y),')

    content_lines.append("}")
    content_lines.append("")

    # Add tuple variable info for log2tslf.py to use
    if tuple_vars:
        content_lines.append("# Tuple variable metadata")
        content_lines.append("TUPLE_VARS = {")
        for tup_name, arity in tuple_vars.items():
            # A tuple is const only if ALL its elements are const
            is_const = all(
                f"{tup_name}[{i}]" in variables and "const" in variables[f"{tup_name}[{i}]"]
                for i in range(arity)
            )
            content_lines.append(f'    "{tup_name}": {{"arity": {arity}, "const": {is_const}}},')
        content_lines.append("}")
        content_lines.append("")

    # Add boolean variable names for log2tslf.py to use
    if boolean_vars:
        content_lines.append("# Boolean stream variables (treated as predicates, not updates)")
        content_lines.append("BOOLEAN_VARS = [")
        for bool_var in sorted(boolean_vars):
            content_lines.append(f'    "{bool_var}",')
        content_lines.append("]")
        content_lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines))
    return out_path


def _iter_function_outputs(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "output_funcs.jsonl":
                yield os.path.join(dirpath, filename)


def gather_functions(root_dir: str):
    """
    Load and deduplicate synthesized functions from a directory tree.
    """
    aggregated_meta: dict[str, tuple[str, Any]] = {}
    aggregated_specs: dict[str, dict[str, Any]] = {}
    seen_keys: set[Any] = set()
    for path in sorted(_iter_function_outputs(root_dir)):
        metadata, specs = extract_function_metadata(path)
        for name, meta in metadata.items():
            spec = specs.get(name)
            if not spec:
                continue
            key = spec.get("key")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            base_name = spec.get("base_name", name)
            unique_name = base_name
            suffix = 2
            while unique_name in aggregated_meta:
                unique_name = f"{base_name}_{suffix}"
                suffix += 1
            spec["name"] = unique_name
            aggregated_meta[unique_name] = meta
            aggregated_specs[unique_name] = spec
    return aggregated_meta, aggregated_specs




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--function_dir", required=True)
    parser.add_argument("--out_dir", required=False)
    parser.add_argument("--pos", help="Explicit path to positive traces directory (default: trace_dir/pos)")
    parser.add_argument("--neg", help="Explicit path to negative traces directory (default: trace_dir/neg)")
    args = parser.parse_args()

    if not args.out_dir :
        args.out_dir = args.function_dir

    assert os.path.exists(args.trace_dir), "Invalid input directory"
    pos_path = args.pos if args.pos else os.path.join(args.trace_dir, "pos")
    neg_path = args.neg if args.neg else os.path.join(args.trace_dir, "neg")
    assert os.path.exists(pos_path) and os.path.exists(neg_path), "Traces not properly bucketed"
    assert os.path.exists(args.function_dir), "Invalid functions directory"

    # GET VARIABLE METADATA
    trace_paths = []
    for path in [pos_path, neg_path]:
        for name in os.listdir(path):
            if not name.endswith(".jsonl"):
                continue
            trace_paths.append(os.path.join(path, name))

    # Detect tuple variables before expansion
    tuple_vars = {}
    for trace_path in trace_paths:
        tuple_vars = get_tuple_vars(trace_path)
        if tuple_vars:
            print(f"Detected tuple variables: {tuple_vars}")
            break

    # Detect boolean variables
    boolean_vars = find_boolean_variables(trace_paths)
    if boolean_vars:
        print(f"Detected boolean variables: {boolean_vars}")

    constants = find_constant_variables(trace_paths)
    variables = {}
    for trace_path in trace_paths:
        variables = extract_vars(trace_path, constants)
        if variables :
            break

    # GET FUNCTION METADATA
    functions, function_specs = gather_functions(args.function_dir)

    print(f"Variables: {variables}")
    print(f"Functions: {functions}")
    print(f"Tuple vars: {tuple_vars}")
    print(f"Boolean vars: {boolean_vars}")

    output_path = write_metadata_file(args.out_dir, variables, function_specs, tuple_vars=tuple_vars, boolean_vars=boolean_vars)
    print(f"Wrote metadata file to {output_path}")


    
