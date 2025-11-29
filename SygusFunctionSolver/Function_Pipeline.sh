# Sample run: ./run_pipeline.sh temp_data output_pairs X Y Z


#!/bin/bash
set -e  # stop on any error

#########################################
# ARGUMENT PARSING
#########################################

if [ "$#" -lt 3 ]; then
    echo "Usage: ./run_pipeline.sh <input_dir> <output_dir> <var1> [var2] [var3] ..."
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2
VARS="$@"   # remaining args are vars

echo "==========================================="
echo " Running Full Pipeline"
echo " Input Dir:  $INPUT_DIR"
echo " Output Dir: $OUTPUT_DIR"
echo " Vars:       $VARS"
echo "==========================================="

#########################################
# STEP 1 — IOSeperation.py
#########################################

echo ""
echo "=== Step 1: IOSeperation.py ==="
python3 IOSeperation.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --vars $VARS

#########################################
# STEP 2 — CreateGroupings.py
#########################################

echo ""
echo "=== Step 2: CreateGroupings.py ==="
python3 CreateGroupings.py \
    --root_dir "$OUTPUT_DIR"

#########################################
# STEP 3 — CreateGroupingSubsets.py
#########################################

echo ""
echo "=== Step 3: CreateGroupingSubsets.py ==="
python3 CreateGroupingSubsets.py \
    --root_dir "$OUTPUT_DIR"

#########################################
# STEP 4 — GenerateFunctions.py
#########################################

echo ""
echo "=== Step 4: GenerateFunctions.py ==="
python3 GenerateFunctions.py \
    --root_dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo " Pipeline Complete!"
echo " Results generated under: $OUTPUT_DIR"
echo "==========================================="
