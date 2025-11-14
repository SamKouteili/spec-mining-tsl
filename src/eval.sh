#!/usr/bin/env bash
set +e
set +o histexpand

original_tsl=$1
tsl_file=$(basename "$original_tsl")
tsl_name="${tsl_file%.*}"
mined_ltl=$2 # NOTE: LTL FOR NOW
mined_dir=$(dirname "$mined_ltl")
samples_path=$3
num_samples=${4:-50}
length=${5:-6}
timeout=${6:-150}

echo "[eval] oTSL:$tsl_name"
echo "[eval] mLTL:$mined_ltl"
echo "[eval] samples_path:$samples_path"
echo "[eval] num_samples:$num_samples"

# mkdir -p "$out_dir"
# rm -rf "$out_dir/*"

if [[ ! -f "$original_tsl" ]]; then
    echo "[eval] Error: Original HOA file $original_hoa not found" >&2
    exit 1
fi

if [[ ! -f "$mined_ltl" ]]; then
    echo "[eval] Error: Mined HOA file $mined_hoa not found" >&2
    exit 1
fi

# samples_path="$out_dir""$tsl_name"p"$num_samples"n"$num_samples"l"$length".trace
echo "[eval] Getting samples for evaluation"
if [[ ! -f "$samples_path" ]]; then
  echo "* Generating samples at $samples_path"
  if ! python tracer.py --tsl "$original_tsl" -o "$samples_path" -p "$num_samples" -n "$num_samples" -l "$length" -t "spot" --timeout "$timeout"; then
    echo "[eval] Error: tracer failed" >&2
    exit 1
  fi
else
  echo "* Using existing samples at $samples_path"
fi

mhoa=$(cat $mined_ltl | ltlf2dfa)

accepted=0
rejected=0
total=0
seen_sep=false

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  line=${line%$'\r'}

  if [[ "$line" == '---' ]]; then
    seen_sep=true
    continue
  fi

  total=$((total+1))

  printf '%s\n' "$mhoa" | python dotomata.py --check "$line" >/dev/null 2>&1
  rc=$?

  if [[ "$seen_sep" == "false" ]]; then
    if [[ $rc -eq 0 ]]; then
      accepted=$((accepted+1))
    fi
  else
    if [[ $rc -ne 0 ]]; then
      rejected=$((rejected+1))
    fi
  fi
done < "$samples_path"

echo "[eval] accepted:$accepted rejected:$rejected total:$total"
accuracy=$(((accepted+rejected)*100/total))
echo "Total Accuracy:$accuracy"
# if [[ $total -eq 0 ]]; then
#   echo "Total Accuracy: N/A (no samples)"
# else``
#   echo "Total Accuracy:$total"
# fi

exit 0


# helper: build a single-quoted literal from $1 (escape any internal safe for bash')
# sq() {
#   # ref: POSIX trick: close ', insert '\'' , reopen '
#   printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
# }

# # optional zero-width space definition
# printf -v _ZWS '\u200b'

# eval_traces() {
#   local hoa_file="$1"
#   local trace_file="$2"
#   local verbose="$3"

#   local acc=0
#   local rej=0

#   while IFS= read -r line || [[ -n "$line" ]]; do
#     # skip blanks or separators
#     [[ -z "$line" || "$line" == '---' ]] && continue

#     # normalize
#     line=${line%$'\r'}
#     line=${line//$_ZWS/}

#     sq_word=$(sq "$line")
#     cmd="autfilt \"$hoa_file\" --accept-word=${sq_word}"

#     stdout_file="$(mktemp)"
#     stderr_file="$(mktemp)"

#     if bash -c "$cmd" >"$stdout_file" 2>"$stderr_file"; then
#       acc=$((acc+1))
#     else
#       rej=$((rej+1))
#       if [[ "$verbose" == "true" ]]; then
#         echo "cmd: $cmd"
#         echo "stdout:"
#         cat "$stdout_file"
#         echo "stderr:"
#         cat "$stderr_file"
#         echo "----"
#       fi
#     fi

#     rm -f "$stdout_file" "$stderr_file"
#   done < "$trace_file"

#   echo "$acc,$rej"
# }

# orig_res=$(eval_traces "$original_hoa" "$out_dir/original.trace" "true")
# mined_res=$(eval_traces "$mined_hoa" "$out_dir/mined.trace" "false")

# echo "[eval] Evaluated $original_hoa on $out_dir/original.trace: $orig_res"
# echo "[eval] Evaluated $mined_hoa on $out_dir/mined.trace: $mined_res"

# # while IFS= read -r line; do
# #     # echo "$line"
# #     if autfilt -v "$original_hoa" --accept-word="$line" >/dev/null; then
# #         echo "✅ Trace is ACCEPTED by original automaton."
# #     else
# #         echo "❌ Trace is REJECTED by original automaton."
# #     fi
# # done < "$out_dir/original.trace"
# # while IFS= read -r line; do
# #     autfilt -v "$mined_hoa" --accept-word "$line"
# # done < "$out_dir/mined.trace"
