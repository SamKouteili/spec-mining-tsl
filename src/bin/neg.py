#!/usr/bin/env python3

import sys
import re

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
        if re.match(r'^\s*always\s+guarantee\s*\{', line):
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

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python neg.py input.tsl output.tsl")
    #     sys.exit(1)

    if sys.argv[1]:


    input_file = sys.argv[1]
    output_file = sys.argv[2]
    negate_tsl(input_file, output_file)