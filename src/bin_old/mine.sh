#!/usr/bin/env bash
set -euo pipefail

tsl_path=$1
out_path=$2
kind=${3:-"scarlet"}  # default to scarlet
num_samples=${4:-10}
length=${5:-5}
sample_bank_path=${6:-}  # trace bank

echo "[mine] tsl: $tsl_path"
echo "[mine] out: $out_path"
echo "[mine] samples: $num_samples"
echo "[mine] length: $length"

tsl_file=$(basename "$tsl_path")
tsl_name="${tsl_file%.*}"

echo "[mine] Running Evaluation Pipeline for $tsl_name..."

if [[ ! -f "$tsl_path" ]]; then
    echo "[tsl] Error: $tsl_path not found" >&2
    exit 1
fi

tsl_file=$(basename "$tsl_path")
tsl_name="${tsl_file%.*}"
tsl_dir=$(dirname "$tsl_path")

out_file=$(basename "$out_path")
out_name="${out_file%.*}"
out_dir=$(dirname "$out_path")

if [[ ! -d "$out_dir" ]]; then
  echo "[mine] Error: Output directory $out_dir does not exist" >&2
  exit 1
fi

echo "[mine] Generating random positive/negative traces"
trace_path="$out_dir"/"$tsl_name"p"$num_samples"n"$num_samples"l"$length".trace # unique trace for mining
if [[ -n "${sample_bank_path:-}" && -f "$sample_bank_path" ]]; then
  echo "* Found sample bank, extracting dataset"
  python utils.py -i "$sample_bank_path" -o "$trace_path" -d "$num_samples"
else
  echo "* Sample bank not found, generating traces"
  python src/tracer.py --tsl "$tsl_path" -o "$trace_path" -p "$num_samples" -n "$num_samples" -l "$length" -t "$kind" --timeout 60
fi

echo "[mine] Mining Specification from Trace"

SECONDS=0
csv_path=$out_dir/$out_name.csv
if [[ "$kind" == "scarlet" ]]; then
  echo "[mine] Mining with scarlet"
  python scarlet.py --tsl $tsl_path  --csv $csv_path -i $trace_path -o $out_path --timeout 7200
elif [[ "$kind" == "bolt" ]]; then
  MAX_SIZE_LTL=10 # Run LTL enumeration until size `max_size_ltl` before switching to a boolean set cover algorithm
  DOMIN_NB=10 # Number of candidates to use for domination checking in the step that converts LTL formulas to boolean formulas
  MAX_SIZE_BOOL=100
  echo "[mine] Mining with bolt"
  bolt $trace_path $MAX_SIZE_LTL $DOMIN_NB enum $MAX_SIZE_BOOL $DOMIN_NB > $out_path.ltl
else 
  echo "[mine] Error: Unsupported mining kind '$kind'. Supported kinds: scarlet, bolt." >&2
  exit 1
fi
duration=$SECONDS
echo $duration > $out_dir/$out_name.time
echo "[mine] Mining completed in $duration seconds"

echo "[mine] Successfully mined specification"

exit 0

# tsl_neg="$tsl_dir/n_$tsl_name.tsl"

# Generate negation
# python3 neg.py "$tsl_path" "$tsl_neg"
# echo "[neg] Generated negation n_$tsl_name for $tsl_name"

# Generate positive and negative HOA
# hoa_path="$tsl_dir/$tsl_name.hoa"
# # n_hoa_path="$tsl_dir/n_$tsl_name.hoa"
# if tsl hoa -i "$tsl_path" > "$hoa_path"; then
#     echo "[hoa] Generated HOA for $tsl_name"
# else
#     echo "[hoa] Error/Warning generating HOA for $tsl_name" >&2
# fi

# if autfilt --complement "$hoa_path" > "$n_hoa_path" && [[ -s "$n_hoa_path" ]]; then
#     echo "[¬hoa] Generated complement HOA for $tsl_name"
# else
#     echo "[¬hoa] Warning: Failed to generate complement HOA for $tsl_name (file may be empty or command failed)" >&2
# fi

# Execute positive and negative hoa
# hoax_dir="$tsl_dir/hoax"
# mkdir -p "$hoax_dir"
# mkdir -p "$hoax_dir/pos"
# mkdir -p "$hoax_dir/neg"

# for ((i=1; i<=num_samples; i++)); do
#     hoax "$hoa_path" --config ../src/TraceGen_Random.toml > "$hoax_dir/pos/$tsl_name"_"$i.hoax"
#     echo "[hoax] Executed HOA $i for $tsl_name"
#     hoax "$n_hoa_path" --config ../src/TraceGen_Random.toml > "$hoax_dir/neg/$tsl_name"_"$i.hoax"
#     echo "[hoax] Executed ¬HOA $i for $tsl_name"
# done
# echo "* Executed HOA $num_samples times for $tsl_name and $tsl_neg"

# Generate positive and negative traces
# trace_path="$tsl_dir/$tsl_name"_"$num_samples.trace"
# python3 hoax.py mine --hoa "$hoa_path" -p "$num_samples" -n "$num_samples" -t scarlet -o "$trace_path"
# echo "[hoax] Executed $num_samples pos/neg traces for $tsl_name"

# # Mine LTL specification
# mined_dir="$tsl_dir/mined"
# tlsf_path="$mined_dir/$tsl_name.tlsf"
# mkdir -p "$mined_dir"
# python3 scarlet.py -i "$trace_path" -o "$tlsf_path" --csvname "$mined_dir/$tsl_name.csv" --timeout 3600
# echo "[scarlet] Mined LTL specification for $tsl_name"

# # Convert LTL specification to TSL

# # python3 tlsf.py "$mined_dir/$tsl_name.csv" "$mined_dir/$tsl_name.tlsf"
# # echo "[tlsf] Retrieved TLSf for $tsl_name"

# # Extract assumptions from original TSL and prepend to mined TSL
# write_assumptions() {
#   local tsl="$1"
#   local out="$2"

#   awk '
#     # portable char counter
#     function count_char(s, ch,   i,n,c) {
#       c=0; n=length(s)
#       for (i=1; i<=n; i++) if (substr(s,i,1)==ch) c++
#       return c
#     }

#     BEGIN { inside=0; depth=0 }
#     { sub(/\r$/, "") }  # strip CR if present

#     # detect a line containing "always assume {"
#     /^[[:space:]]*always[[:space:]]+assume[[:space:]]*{/ {
#       inside=1
#       print
#       depth += count_char($0, "{") - count_char($0, "}")
#       if (depth <= 0) { inside=0; depth=0 }
#       next
#     }

#     # while inside a block, print lines and track brace depth
#     inside {
#       print
#       depth += count_char($0, "{") - count_char($0, "}")
#       if (depth <= 0) { inside=0; depth=0 }
#       next
#     }
#   ' "$tsl" > "$out"
# }

# write_assumptions "$tsl_path" "$mined_dir/$tsl_name.tsl"
# # echo "[tsl] Extracted assumptions from original TSL for $tsl_name"
# tsl fromtlsf -i "$tlsf_path" >> "$mined_dir/$tsl_name.tsl"
# echo "[tsl] Converted TLSf to TSL. Final formula:"
# tsl hoa -i "$mined_dir/$tsl_name.tsl" > "$mined_dir/$tsl_name.hoa"
# # echo "[hoa] Generated HOA for mined TSL"

# cat "$mined_dir/$tsl_name.tsl"
# # head -n 50 "$tsl_dir/$tsl_name.TSL"

# exit 0
