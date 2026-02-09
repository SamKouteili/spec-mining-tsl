#!/usr/bin/env bash

for dir in ../tslf/*/; do
    benchmark=$(basename "$dir")
    tsl_file="../tslf/$benchmark/$benchmark.tsl"

    if [[ -f "$tsl_file" ]]; then
        echo "=== $benchmark ==="
        cat "$tsl_file" | tsl tlsf | python utils.py -e ASSUME | ltlfilt --simplify
        cat "$tsl_file" | tsl tlsf | python utils.py -e ASSUME | ltlfilt --format='%[v]h'
        echo ""
    fi
done
