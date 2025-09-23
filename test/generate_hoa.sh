#!/usr/bin/env bash
set -euo pipefail

# Script to generate HOA files from TSL files with same directory structure

# Create test/hoa directory if it doesn't exist
mkdir -p test/hoa

# Find all directories in test/tsl
find tsl -mindepth 1 -maxdepth 1 -type d | while read -r tsl_dir; do
    # Extract directory name
    dir_name=$(basename "$tsl_dir")

    # Create corresponding directory in test/hoa
    mkdir -p "hoa/$dir_name"

    # Process all .tsl files in this directory
    find "$tsl_dir" -name "*.tsl" -type f | while read -r tsl_file; do
        # Extract filename without path and extension
        filename=$(basename "$tsl_file" .tsl)

        # Generate HOA for the TSL file
        hoa_file="hoa/$dir_name/$filename.hoa"
        echo "Converting $tsl_file to $hoa_file"
        if tsl hoa -i "$tsl_file" > "$hoa_file" 2>&1; then
            echo "Generated HOA for $filename in $dir_name"
        else
            echo "Warning: Failed to generate HOA for $filename in $dir_name"
        fi
    done
done

echo "All TSL to HOA conversions completed successfully!"
