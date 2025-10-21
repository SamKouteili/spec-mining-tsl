#!/usr/bin/env python3

import csv
import sys
import os
import argparse

def extract_last_formula(input_file):
    """Extract the last completed formula from scarlet_out.csv"""

    if not os.path.isfile(input_file):
        print(f"Error: Input file {input_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip header, find last row with 4 columns and last column is '1' (completed)
    for row in reversed(rows[1:]):  # Skip header
        if len(row) >= 4 and row[3].strip() == '1':
            formula = row[2].strip()
            return formula

    print('Error: No completed formula found', file=sys.stderr)
    sys.exit(1)

def write_formula_to_file(formula, output_file):
    """Write formula wrapped in GUARANTEE block to output file"""

    content = f"""GUARANTEE {{
    {formula};
}}"""

    with open(output_file, 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='Extract the last formula from scarlet_out.csv and wrap it')
    parser.add_argument('input_file', nargs='?', default='src/scarlet_out.csv',
                        help='Input CSV file (default: src/scarlet_out.csv)')
    parser.add_argument('output_file', nargs='?', default='mined_formula.txt',
                        help='Output file (default: mined_formula.txt)')

    args = parser.parse_args()

    # Extract the last completed formula
    last_formula = extract_last_formula(args.input_file)

    # Write to output file wrapped with GUARANTEE
    write_formula_to_file(last_formula, args.output_file)

    # Optional: uncomment these lines to see output
    # print(f"Extracted formula written to {args.output_file}:")
    # print(f"Formula: {last_formula}")

if __name__ == "__main__":
    main()