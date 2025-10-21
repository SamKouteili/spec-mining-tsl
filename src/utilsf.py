import sys
import re
from pathlib import Path
from typing import Optional, Tuple
import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=False, help="Input file path")
    parser.add_argument("-o", type=str, required=False, help="Output file path")
    parser.add_argument("-e", type=str, help="Extract assumptions")
    parser.add_argument("-n", action="store_true", help="Negate input tsl")
    return parser.parse_args()


def get_input() -> str:
    """Return command line input until EOF"""
    read = []
    while True:
        try:
            l = input()
        except EOFError:
            break
        if l == "":
            break
        read.append(l)
    return "\n".join(read)

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

# def extract_assumptions(text: str) -> str:
#     """Extract and normalise the assumptions contained within curly braces.

#     Deprecated: Use extract_block(text, "assume") instead.
#     """
#     return extract_block(text, "assume")


if __name__ == "__main__":
    args = parser()
    args = vars(args)

    text = get_input() if not args.get('i') else Path(args['i']).read_text()
    # print(text)

    out = ""
    if args.get("e"):
        block = extract_block(text, args["e"])
        out += " & ".join([a.strip("\n") for a in block.split(";") if a.strip("\n")])
    if args.get("n"):
        out += negate_tsl(text)

    if args['o']:
        with open(args['o'], 'w') as f:
            f.write(out)
    else :
        print(out)