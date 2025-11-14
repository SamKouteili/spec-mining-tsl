#!/usr/bin/env python3
import re
import sys
import subprocess

if len(sys.argv) < 3:
    print("Usage: python check_trace.py <hoa_file> <ap1> <ap2> ...")
    sys.exit(1)

hoa_file = sys.argv[1]
aps = sys.argv[2:]

trace = []

# Regex matches either set() or {…} inside the (state, {APs}, state)
pattern = re.compile(r"\(\s*\d+,\s*(set\(\)|\{.*?\}),\s*\d+\)")

with open("//hoax_raw.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue

        brace_content = m.group(1).strip()

        # Handle empty cases
        if brace_content in ("set()", "{}"):
            present = set()
        else:
            # remove outer { }, split by comma, strip quotes
            inner = brace_content.strip("{}").strip()
            if inner:
                present = {tok.strip().strip("'\"") for tok in inner.split(",") if tok.strip()}
            else:
                present = set()

        # Build valuation string for this time step
        assignment = [ap if ap in present else "!" + ap for ap in aps]
        trace.append("&".join(assignment))

# --- Build Spot word ---
spot_word = ";".join(trace) + ";cycle{1}"
spot_word = spot_word.strip()  # Remove any accidental newlines/whitespace

# Print debug info
print("📜 Reconstructed trace valuations:")
for i, t in enumerate(trace):
    print(f"  step {i}: {t}")

print("\n🔎 Spot word repr (for debugging hidden chars):")
print(repr(spot_word))
print()

# Save to file
trace_file = "./trace.txt"
with open(trace_file, "w") as f:
    f.write(spot_word + "\n")

print(f"📄 Spot trace written to {trace_file}:")
print(spot_word)

# Run autfilt with -v to see state path
print("\n🔍 Checking trace with autfilt...")
try:
    result = subprocess.run(
        ["autfilt", "-v", hoa_file, "--accept-word", spot_word],
        check=False,
        text=True,
        capture_output=True
    )
    # Print autfilt's output (state path + acceptance info)
    if result.stdout.strip():
        print("\n--- autfilt stdout ---")
        print(result.stdout)
    if result.stderr.strip():
        print("\n--- autfilt stderr ---")
        print(result.stderr)

    if result.returncode == 0:
        print("✅ Trace is ACCEPTED by automaton.")
    else:
        print("❌ Trace is REJECTED by automaton.")
except FileNotFoundError:
    print("Error: autfilt not found in PATH. Install Spot and make sure autfilt is accessible.")
