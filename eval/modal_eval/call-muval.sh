#!/bin/bash
# MuVal wrapper for issy --pruning 2
# Reads HES from stdin, calls MuVal, outputs result

LOGFILE="/tmp/muval_calls.log"
OLDDIR=$(pwd)
INFILE=$(mktemp --suffix=.hes)
OPTS="-c ./config/solver/dbg_muval_parallel_exc_tb_ar.json -p muclp"
TIMEOUT=$1

# Read input from stdin
tee > $INFILE

# Log to both stderr AND a file (so we can verify it was called)
echo "=============================================" >> $LOGFILE
echo "[call-muval] $(date): INVOKED" >> $LOGFILE
echo "[call-muval] Timeout: $TIMEOUT seconds" >> $LOGFILE
echo "[call-muval] Input file: $INFILE" >> $LOGFILE
echo "[call-muval] Input size: $(wc -c < $INFILE) bytes" >> $LOGFILE

# Also print to stderr for immediate visibility
echo "[call-muval] *** MUVAL INVOKED *** timeout=$TIMEOUT" >&2

cd /muval

# Set up OPAM environment
export OPAMROOT=/root/.opam
eval $(opam env --root=/root/.opam --switch=5.2.0) 2>/dev/null

# Check if MuVal binary exists
if [ ! -f "./_build/default/main.exe" ]; then
    echo "[call-muval] ERROR: MuVal binary not found!" | tee -a $LOGFILE >&2
    rm -f $INFILE
    exit 1
fi

echo "[call-muval] Running MuVal..." >> $LOGFILE
START_TIME=$(date +%s)

# Run MuVal
timeout $TIMEOUT ./_build/default/main.exe $OPTS $INFILE 2>> $LOGFILE
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "[call-muval] MuVal finished in ${ELAPSED}s with exit code $EXIT_CODE" >> $LOGFILE
echo "[call-muval] *** MUVAL DONE *** exit=$EXIT_CODE time=${ELAPSED}s" >&2

rm -f $INFILE
cd $OLDDIR
exit $EXIT_CODE
