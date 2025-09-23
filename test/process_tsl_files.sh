#!/usr/bin/env bash
set -euo pipefail

# Script to regenerate negated TSL files in existing subdirectories

# Find all subdirectories in tsl
find tsl -mindepth 1 -maxdepth 1 -type d | while read -r tsl_dir; do
    # Extract directory name
    dir_name=$(basename "$tsl_dir")

    # Look for the original TSL file (should match directory name)
    original_file="$tsl_dir/$dir_name.tsl"

    if [[ -f "$original_file" ]]; then
        # Generate negation using neg.sh and save as n_<name>.tsl
        negated_file="$tsl_dir/n_$dir_name.tsl"
        python3 ../src/neg.py "$original_file" "$negated_file"
        echo "Regenerated negation for $dir_name"
    else
        echo "Warning: Original file $original_file not found in $dir_name"
    fi
done

echo "All negated TSL files regenerated successfully!"
