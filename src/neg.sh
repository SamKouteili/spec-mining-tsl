#!/usr/bin/env bash
set -euo pipefail

infile="$1"
outfile="$2"

awk '
  BEGIN { in_g = 0; clause = ""; depth_p = 0; depth_b = 0 }

  function update_depths(s,   i,c) {
    for (i=1;i<=length(s);i++) {
      c = substr(s,i,1)
      if (c=="(") depth_p++
      else if (c==")") depth_p--
      else if (c=="[") depth_b++
      else if (c=="]") depth_b--
    }
  }

  function emit_clause(raw,   s) {
    s = raw
    sub(/[ \t]*;[ \t]*$/, "", s)  # strip final semicolon
    print "    !(" s ");"
  }

  /^[[:space:]]*always[[:space:]]+guarantee[[:space:]]*{/ {
    print; in_g=1; clause=""; depth_p=0; depth_b=0; next
  }

  in_g && /^[[:space:]]*}/ {
    if (clause != "") { emit_clause(clause); clause="" }
    print; in_g=0; next
  }

  {
    if (!in_g) { print; next }

    line=$0
    if (clause == "") clause=line
    else clause=clause "\n" line

    update_depths(line)

    if (line ~ /;[[:space:]]*$/ && depth_p==0 && depth_b==0) {
      emit_clause(clause)
      clause=""
    }
    next
  }
' "$infile" > "$outfile"
