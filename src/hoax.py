# run_hoax.py
import subprocess
import sys

if len(sys.argv) < 3:
    print("Usage: python run_hoax.py <hoa_file> <out_file>")
    sys.exit(1)

hoa_file = sys.argv[1]
out_file = sys.argv[2]
config_file = "./TraceGen_Random.toml"


try:
    result = subprocess.run(
        ["hoax", hoa_file, "--config", config_file],
        check=True,
        text=True,
        capture_output=True
    )
except subprocess.CalledProcessError as e:
    print("Error running hoax:")
    print(e.stderr)
    sys.exit(1)

with open(out_file, "w") as f:
    f.write(result.stdout)

print(f"Hoax output written to {out_file}")
