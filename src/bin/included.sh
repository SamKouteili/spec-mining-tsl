#!/usr/bin/env bash

# check_inclusion A.hoa B.hoa  -> checks L(A) ⊆ L(B) and L(B) ⊆ L(A)
A="$1"; B="$2"
if autfilt --included-in="$B" "$A" >/dev/null; then
  echo "L($A) ⊆ L($B)"
else
  echo "L($A) ⊄ L($B)"
fi
if autfilt --included-in="$A" "$B" >/dev/null; then
  echo "L($B) ⊆ L($A)"
else
  echo "L($B) ⊄ L($A)"
fi
