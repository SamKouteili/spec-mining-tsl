#!/bin/bash
set -e  # stop on any error

#########################################
# ARGUMENT PARSING
#########################################

if [ "$#" != 1 ]; then
    echo "Usage: ./functions.sh <input_dir> "
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${INPUT_DIR}/out"

sh functions.sh "$INPUT_DIR" "$OUTPUT_DIR"
sh bolt.sh "$INPUT_DIR" "$OUTPUT_DIR/metadata.py" "$OUTPUT_DIR"

