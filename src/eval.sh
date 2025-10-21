#!/usr/bin/env bash
set +e
set +o histexpand

original_hoa=$1
hoa_file=$(basename "$original_hoa")
hoa_name="${hoa_file%.*}"
mined_hoa=$2
mined_dir=$(dirname "$mined_hoa")
out_dir=$3

mkdir -p "$out_dir"

rm -rf "$out_dir/*"

num_samples=${4:-20}

if [[ ! -f "$original_hoa" ]]; then
    echo "Error: Original HOA file $original_hoa not found" >&2
    exit 1
fi

if [[ ! -f "$mined_hoa" ]]; then
    echo "Error: Mined HOA file $mined_hoa not found" >&2
    exit 1
fi

echo "Original HOA: $original_hoa"
echo "Mined HOA: $mined_hoa"
echo "Out dir: $out_dir"
echo "Num samples: $num_samples"

# hoax_dir="$out_dir/hoax"
# mkdir -p "$hoax_dir/pos"

# for ((i=1; i<=num_samples; i++)); do
#     hoax "$original_hoa" --config ../src/TraceGen_Random.toml > "$hoax_dir/pos/$hoa_name"_"$i.hoax"
# done
# echo "[hoax] Executed $num_samples hoax execution traces for $original_hoa"

python3 hoax.py eval --hoa "$original_hoa" --hoam "$mined_hoa" -p "$num_samples" -n "$num_samples" -o "$out_dir/original.trace" -t spot
echo "[trace] Generated spot-like traces from hoax for $original_hoa"

# python3 hoax.py eval --hoa "$mined_hoa" -p "$num_samples" -n "$num_samples" -o "$out_dir/mined.trace" -t spot
# echo "[trace] Generated spot-like traces from hoax for $mined_hoa"

exit 0

# helper: build a single-quoted literal from $1 (escape any internal safe for bash')
sq() {
  # ref: POSIX trick: close ', insert '\'' , reopen '
  printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

# optional zero-width space definition
printf -v _ZWS '\u200b'

eval_traces() {
  local hoa_file="$1"
  local trace_file="$2"
  local verbose="$3"

  local acc=0
  local rej=0

  while IFS= read -r line || [[ -n "$line" ]]; do
    # skip blanks or separators
    [[ -z "$line" || "$line" == '---' ]] && continue

    # normalize
    line=${line%$'\r'}
    line=${line//$_ZWS/}

    sq_word=$(sq "$line")
    cmd="autfilt \"$hoa_file\" --accept-word=${sq_word}"

    stdout_file="$(mktemp)"
    stderr_file="$(mktemp)"

    if bash -c "$cmd" >"$stdout_file" 2>"$stderr_file"; then
      acc=$((acc+1))
    else
      rej=$((rej+1))
      if [[ "$verbose" == "true" ]]; then
        echo "cmd: $cmd"
        echo "stdout:"
        cat "$stdout_file"
        echo "stderr:"
        cat "$stderr_file"
        echo "----"
      fi
    fi

    rm -f "$stdout_file" "$stderr_file"
  done < "$trace_file"

  echo "$acc,$rej"
}

orig_res=$(eval_traces "$original_hoa" "$out_dir/original.trace" "true")
mined_res=$(eval_traces "$mined_hoa" "$out_dir/mined.trace" "false")

echo "[eval] Evaluated $original_hoa on $out_dir/original.trace: $orig_res"
echo "[eval] Evaluated $mined_hoa on $out_dir/mined.trace: $mined_res"

# while IFS= read -r line; do
#     # echo "$line"
#     if autfilt -v "$original_hoa" --accept-word="$line" >/dev/null; then
#         echo "✅ Trace is ACCEPTED by original automaton."
#     else
#         echo "❌ Trace is REJECTED by original automaton."
#     fi
# done < "$out_dir/original.trace"
# while IFS= read -r line; do
#     autfilt -v "$mined_hoa" --accept-word "$line"
# done < "$out_dir/mined.trace"
