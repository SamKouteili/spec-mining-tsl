# Sample run: ./run_pipeline.sh temp_data output_pairs X Y Z


#!/bin/bash
set -e  # stop on any error

#########################################
# ARGUMENT PARSING
#########################################

if [ "$#" != 2 ]; then
    echo "Usage: ./functions.sh <input_dir> <output_dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

echo "==========================================="
echo " Running Full Pipeline"
echo " Input Dir:  $INPUT_DIR"
echo " Output Dir: $OUTPUT_DIR"
echo "==========================================="

#########################################
# STEP 1 — IOSeperation.py
#########################################

echo ""
echo "=== Step 1: IOSeperation.py ==="
python3 SygusFunctionSolver/IOSeperation.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

#########################################
# STEP 2 — CreateGroupings.py
#########################################

echo ""
echo "=== Step 2: CreateGroupings.py ==="
python3 SygusFunctionSolver/CreateGroupings.py \
    --root_dir "$OUTPUT_DIR"

# exit 0

#########################################
# STEP 3 — CreateGroupingSubsets.py
#########################################

echo ""
echo "=== Step 3: CreateGroupingSubsets.py ==="
python3 SygusFunctionSolver/CreateGroupingSubsets.py \
    --root_dir "$OUTPUT_DIR"

#########################################
# STEP 4 — GenerateFunctions.py
#########################################

echo ""
echo "=== Step 4: GenerateFunctions.py ==="
python3 SygusFunctionSolver/GenerateFunctions.py \
    --root_dir "$OUTPUT_DIR"

echo "=== Step 4: CreateMetadata.py ==="
python3 SygusFunctionSolver/CreateMetadata.py \
    --trace_dir "$INPUT_DIR" \
    --function_dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo " Pipeline Complete!"
echo " Results generated under: $OUTPUT_DIR"
echo "==========================================="
