#!/usr/bin/env python3

import sys
import re
from collections import OrderedDict

def parse_hoax_trace(filename):
    """Parse hoax output and convert to spot format"""

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract trace steps (ignore hook/bound messages)
    trace_steps = []
    propositions = set()

    for line in lines:
        line = line.strip()
        if not line or 'Hook:' in line or 'Bound:' in line:
            continue

        # Parse format: state: (current_state, {prop_set}, next_state)
        # Note: first number is always 0, actual step is line order
        match = re.match(r'\d+:\s*\(\d+,\s*\{([^}]*)\},\s*\d+\)', line)
        if match:
            step_num = len(trace_steps)  # Use line order as step number
            props_str = match.group(1).strip()

            # Parse propositions
            step_props = set()
            if props_str:
                # Split by comma and clean up
                for prop in props_str.split(','):
                    prop = prop.strip().strip("'\"")
                    if prop:
                        step_props.add(prop)
                        propositions.add(prop)

            trace_steps.append((step_num, step_props))

    return trace_steps, sorted(propositions)

def decode_proposition(prop):
    """Decode hoax proposition names back to readable form"""
    # Remove prefixes and decode separators
    prop = re.sub(r'^p0[bp]0', '', prop)  # Remove p0b0 or p0p0 prefix
    prop = prop.replace('29', '.')  # 29 seems to be separator for dots
    prop = prop.replace('0', ' ')   # 0 seems to be separator for spaces
    prop = prop.replace(' ', '').replace('..', '.')  # Clean up
    return prop

def convert_to_spot_format(trace_steps, propositions, output_file):
    """Convert parsed trace to spot format"""

    # Decode proposition names
    decoded_props = [decode_proposition(prop) for prop in propositions]
    prop_mapping = dict(zip(propositions, decoded_props))

    print(f"Propositions found: {decoded_props}")

    # Build binary trace
    trace_line = []
    for step_num, step_props in trace_steps:
        # Create binary vector for this step
        binary_step = []
        for prop in propositions:
            binary_step.append('1' if prop in step_props else '0')
        trace_line.append(','.join(binary_step))

    # Write to file
    with open(output_file, 'w') as f:
        # Write the single trace (positive example)
        f.write(';'.join(trace_line) + '\n')
        # Add separator for negative examples (empty for now)
        f.write('---\n')

    # Save metadata file with alphabet mapping
    metadata_file = output_file.replace('.trace', '_metadata.json')
    metadata = {
        'original_hoax_props': propositions,
        'decoded_props': decoded_props,
        'prop_mapping': prop_mapping,
        'scarlett_alphabet': {f'p{i}': decoded_props[i] for i in range(len(decoded_props))},
        'proposition_order': decoded_props,
        'trace_length': len(trace_steps),
        'num_propositions': len(propositions)
    }

    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Converted trace written to {output_file}")
    print(f"Metadata written to {metadata_file}")
    print(f"Proposition order: {decoded_props}")
    print(f"Scarlett will use alphabet: {list(metadata['scarlett_alphabet'].keys())}")

    return metadata

def main():
    if len(sys.argv) != 3:
        print("Usage: python hoax_to_spot.py input_hoax_trace output_spot_trace")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    trace_steps, propositions = parse_hoax_trace(input_file)
    print(f"Parsed {len(trace_steps)} steps with {len(propositions)} propositions")

    convert_to_spot_format(trace_steps, propositions, output_file)

if __name__ == "__main__":
    main()