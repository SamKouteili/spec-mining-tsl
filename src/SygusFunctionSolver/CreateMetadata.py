import os
import json
import argparse
from typing import Any
# from IOSeperation import extract_first_line_keys

OP_PREFIX = {
    "+": "add",
    "*": "mul",
    "-": "sub",
}

def clean_line(raw):
    return (
        raw.replace("\ufeff", "")
           .replace("\xa0", " ")
           .strip()
    )

def extract_vars(file_path) -> dict[str, str]:
    """
    Return a dictionary of variables in the logs and their types
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
                for k in obj.keys():
                    d[k] = type(obj[k]).__name__
            return d
    return d


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


def extract_function_metadata(functions_path: str, *, include_specs: bool = False):
    """
    Convert solver output functions into callable Python metadata.
    """
    with open(functions_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = payload.get("functions", [])
    if not isinstance(entries, list):
        return {}

    metadata: dict[str, tuple[str, Any]] = {}
    specs: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, str):
            continue
        block = _extract_define_block(raw)
        if not block:
            continue
        try:
            parsed = _parse_define_fun(block)
        except ValueError:
            continue
        if not parsed or parsed[0] != "define-fun":
            continue
        args_section = parsed[2] if len(parsed) > 2 else []
        arg_names = [arg[0] for arg in args_section if isinstance(arg, list) and arg]
        expr = parsed[4] if len(parsed) > 4 else None
        if expr is None:
            continue
        try:
            base_name = _build_function_name(expr)
        except Exception as exc:
            print(
                f"[CreateMetadata] Skipping function in {functions_path}: "
                f"{exc}"
            )
            continue
        unique_name = base_name
        counter = 2
        while unique_name in metadata:
            unique_name = f"{base_name}_{counter}"
            counter += 1
        try:
            python_expr = _build_python_expression(expr)
            impl = _build_lambda(arg_names, python_expr)
        except ValueError as exc:
            print(
                f"[CreateMetadata] Unable to convert function in {functions_path}: "
                f"{exc}"
            )
            continue
        arity = len(arg_names)
        sig = _render_type(arity)
        metadata[unique_name] = (sig, impl)
        if include_specs:
            canonical_key = _canonical_key(expr, arg_names, sig)
            specs[unique_name] = {
                "type": sig,
                "args": arg_names,
                "expr": python_expr,
                "key": canonical_key,
                "base_name": base_name,
            }
    if include_specs:
        return metadata, specs
    return metadata


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
    lines = ["FUNCTIONS = {"]
    for name in sorted(function_specs.keys()):
        spec = function_specs[name]
        args = spec.get("args", [])
        lambda_args = ", ".join(args) if args else "_"
        expr = spec.get("expr", "0")
        type_sig = spec.get("type", "int->int")
        lines.append(
            f'    "{name}": ("{type_sig}", lambda {lambda_args}: {expr}),'
        )
    if len(lines) == 1:
        lines.append("    # No functions synthesized")
    lines.append("}")
    return lines


def write_metadata_file(out_dir: str, variables: dict[str, str], function_specs: dict[str, dict[str, Any]]):
    """
    Write VARS and FUNCTIONS dictionaries matching the f/ball/ball.py style.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metadata.py")
    content_lines: list[str] = []
    content_lines.extend(_format_vars_block(variables))
    content_lines.append("")
    content_lines.extend(_format_functions_block(function_specs))
    content_lines.append("")
    content_lines.append("PREDICATES = {}")
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
        metadata, specs = extract_function_metadata(path, include_specs=True)
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
    args = parser.parse_args()

    if not args.out_dir :
        args.out_dir = args.function_dir

    assert os.path.exists(args.trace_dir), "Invalid input directory"
    pos_path = os.path.join(args.trace_dir, "pos")
    # neg_path = os.path.join(input_dir, "neg")
    assert os.path.exists(pos_path), "Traces not properly bucketed"
    assert os.path.exists(args.function_dir), "Invalid functions directory"

    # GET VARIABLE METADATA
    variables = {}
    for trace in os.listdir(pos_path):
        if not trace.endswith(".jsonl"):
            continue
        trace_path = os.path.join(pos_path, trace)
        variables = extract_vars(trace_path)
        if variables :
            break
    
    # GET FUNCTION METADATA
    functions, function_specs = gather_functions(args.function_dir)

    print(variables)

    print(functions)

    output_path = write_metadata_file(args.out_dir, variables, function_specs)
    print(f"Wrote metadata file to {output_path}")


    
