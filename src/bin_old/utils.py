import re
from pathlib import Path
from typing import Optional, Tuple
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=False, help="Input file path")
    parser.add_argument("-o", type=str, required=False, help="Output file path")
    parser.add_argument("-e", type=str, help="Extract block")
    parser.add_argument("-n", type=str, choices=["tsl", "tlsf"], help="Negate input of tsl or tlsf")
    parser.add_argument("-d", type=int, help="Number of entries to extract from database")
    return parser.parse_args()


def get_input() -> str:
    """Return command line input until EOF"""
    read = []
    while True:
        try:
            l = input()
        except EOFError:
            break
        # if l == "":
        #     break
        read.append(l)
    return "\n".join(read)


def get_ap_list(hoa: str) -> list[str]:
    """Extract the atomic propositions declared in a HOA."""
    for line in hoa.splitlines():
        if line.strip().startswith("AP:"):
            # print("APs:\n", line, "\n\n")
            tokens = line.split()
            if len(tokens) < 3:
                print("AP line should be formatted 'AP: <count> <proposition1> <proposition2> ...'")
                return []
            # tokens[0] == "AP:", tokens[1] == count, rest are proposition names
            return [token.strip('"').strip() for token in tokens[2:]]
    return []

def _extract_block(text: str, start_index: int) -> Tuple[Optional[str], int]:
    """Return the contents of the first balanced brace block starting at start_index."""
    brace_start = text.find("{", start_index)
    if brace_start == -1:
        return None, start_index

    depth = 0
    for idx in range(brace_start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : idx], idx + 1
    return None, len(text)


def extract_block(text: str, block_type: str) -> str:
    """Extract and normalise blocks of a specific type contained within curly braces.

    Args:
        text: TSL or TLSF source text
        block_type: Type of block to extract (e.g., "assume", "guarantee")

    Returns:
        Concatenated contents of all matching blocks
    """
    # Match optional "always" or "initially" prefix, then block_type
    # Examples: "assume", "always assume", "initially guarantee", "GUARANTEE"
    block_pattern = re.compile(
        rf"\b(?:always\s+|initially\s+)?{re.escape(block_type)}\b",
        re.IGNORECASE
    )

    blocks: list[str] = []
    search_pos = 0
    while match := block_pattern.search(text, search_pos):
        block, next_pos = _extract_block(text, match.end())
        search_pos = next_pos
        if block is None:
            continue
        lines = [line.strip() for line in block.splitlines()]
        if lines:
            blocks.append("\n".join(lines))

    return "".join(blocks)


def negate_tsl(tsl):
    # with open(input_file, 'r') as f:
    #     content = f.read()

    # Split into lines for processing
    lines = tsl.split('\n')
    result_lines = []

    in_guarantee = False
    guarantee_content = []

    for line in lines:
        # Check if we're entering the guarantee section
        if re.match(r'^\s*(?:always|initially)\s+guarantee\s*\{', line, re.IGNORECASE):
            in_guarantee = True
            result_lines.append(line)
            continue

        # Check if we're exiting the guarantee section
        if in_guarantee and re.match(r'^\s*\}', line):
            # Process collected guarantee content
            if guarantee_content:
                # Join all guarantee content
                full_guarantee = '\n'.join(guarantee_content)
                # Remove trailing semicolons and whitespace
                full_guarantee = re.sub(r';\s*$', '', full_guarantee.strip())

                if full_guarantee:
                    # Check if we have multiple clauses (semicolons followed by actual content)
                    if re.search(r';\s*\n\s*\S', full_guarantee):
                        # Multiple clauses: split on semicolons and join with &&
                        clauses = [clause.strip() for clause in full_guarantee.split(';') if clause.strip()]
                        combined = ' && '.join(clauses)
                        result_lines.append(f"    !({combined});")
                    else:
                        # Single clause: just negate it
                        result_lines.append(f"    !({full_guarantee});")

            result_lines.append(line)
            in_guarantee = False
            guarantee_content = []
            continue

        # If we're in guarantee section, collect content
        if in_guarantee:
            guarantee_content.append(line)
        else:
            # Not in guarantee section, pass through unchanged
            result_lines.append(line)

    # Write result
    # if output_file:
    #     with open(output_file, 'w') as f:
    #         f.write('\n'.join(result_lines))

    return "\n".join(result_lines)


def negate_tlsf(tlsf: str) -> str:
    """Negate a TLSF specification by negating only the last statement in GUARANTEE.

    The last statement is assumed to be the actual specification, while earlier
    statements are mutex constraints on variable updates that should not be negated.

    Args:
        tlsf: TLSF specification string

    Returns:
        Modified TLSF with the last guarantee statement negated
    """
    # Match GUARANTEE block (case-insensitive)
    guarantee_pattern = re.compile(
        r'(\s*GUARANTEE\s*\{)(.*?)(\})',
        re.IGNORECASE | re.DOTALL
    )

    def negate_last_statement(match):
        opening = match.group(1)  # "GUARANTEE {"
        content = match.group(2)   # Everything inside braces
        closing = match.group(3)   # "}"

        # Split on semicolons to get individual statements
        statements = [s.strip() for s in content.split(';') if s.strip()]

        if not statements:
            # No statements found, return unchanged
            return match.group(0)

        # Negate only the last statement by prepending !
        statements[-1] = f"!({statements[-1]})"

        # Reconstruct the GUARANTEE block
        # Preserve original formatting by joining with semicolons and newlines
        reconstructed = ';\n    '.join(statements)
        return f"{opening}\n    {reconstructed};\n{closing}"

    # Replace the GUARANTEE block
    result = guarantee_pattern.sub(negate_last_statement, tlsf)

    return result

def sub_dataset_scarlet(scarlet_data: str, i: int) -> str:
    # pos, neg, ops, aps = [s.strip("\n") for s in scarlet_data.split("---")]
    s = [s.strip("\n") for s in scarlet_data.split("---")]
    # print("S", s)
    pos, neg, ops, aps = [s.strip("\n") for s in scarlet_data.split("---")]
    # print(pos)
    # print(neg)
    # print(ops)
    # print(aps)
    return f"{"\n".join(pos.split("\n")[:i])}\n---\n{"\n".join(neg.split("\n")[:i])}\n---\n{ops}\n---\n{aps}"


if __name__ == "__main__":
    args = parser()
    args = vars(args)

    assert not (args.get("e") and args.get("n") and args.get("d")), "Cannot extract block and negate and dataset"

    text = get_input() if not args.get('i') else Path(args['i']).read_text()
    # print(text)
    # print(text)

    out = ""
    if args.get("e"):
        block = extract_block(text, args["e"])
        out += " & ".join([a.strip("\n") for a in block.split(";") if a.strip("\n")])
    if args.get("n"):
        if args["n"] == "tsl":
            out += negate_tsl(text)
        elif args["n"] == "tlsf":
            out += negate_tlsf(text)
        else:
            print("Invalid format to negate...")
            exit(1)
    if args.get("d"):
        num = args["d"]
        assert num > 0, "Cannot extract non positive number of samples"
        out += sub_dataset_scarlet(text, num)


    if args['o']:
        with open(args['o'], 'w') as f:
            f.write(out)
    else :
        print(out)