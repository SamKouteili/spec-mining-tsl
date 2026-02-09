#!/bin/bash
/opt/bin/issy-bin \
    --caller-z3 /opt/bin/z3 \
    --caller-aut /opt/bin/ltl2tgba \
    --caller-muval /opt/bin/call-muval \
    "$@"
